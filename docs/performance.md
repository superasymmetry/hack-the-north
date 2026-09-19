# Making JarvisVLA fast enough to watch

JarvisVLA started this session at **1.6 environment steps per second**, against a Minecraft
that runs at 20. This is the record of where that time was going, what was done about it, and
what each fix cost in behaviour — because two of them are not free.

The short version: almost none of it was the model thinking. It was the network under the
model, and then, once the network was as good as it was going to get, the fact that the game
was standing still while it waited.

## The setup, and why it is unusual

The policy is a 7B vision-language model. It does not fit on the laptop next to Minecraft, so
it runs under vLLM on an Oscar compute node, and the agent reaches it through an SSH tunnel
that hops via the login node. The client is called **once per environment step** — the sim
takes one tick, the frame goes up, one action comes back — so a rollout is a strictly serial
chain of round trips. That is the property that makes this workload unlike almost anything
else you would point at an LLM server: latency per call matters far more than throughput, and
there is never more than one request in flight.

Each call carries three JPEG frames (the current observation plus two of history), base64
encoded into a chat message, and gets back about eight tokens.

## Where the time actually went

The first thing worth knowing is that the question "is it inference or is it the round trip?"
has a clean answer, and it is not the one you would guess from the model's size.

| where a step went | ms | share |
|---|---:|---:|
| round-trip floor (tunnel + HTTP, one token in, one out) | 95 | 15% |
| uploading 127 KB of base64 JPEG | 350 | 55% |
| opening a fresh TCP connection — a new SSH channel — per call | 110 | 17% |
| GPU prefill + decode (968 prompt tokens, 8 generated) | 130 | 20% |
| **total** | **~640** | 1.6 steps/s |

The GPU was busy for a fifth of each step and idle for the rest.

Getting that breakdown needed a way to price each layer separately, which is worth writing
down because it generalises to any hosted-model client:

- **Weighing the bytes on their own.** A POST with `"model": "nope"` is read in full by the
  server and then rejected before any model work happens. The time it takes is upload plus
  parse, with inference excluded by construction.
- **Separating prefill from decode.** The same real prompt with `max_tokens=1` pays prefill
  but almost no decode; the difference against a full generation is the decode.
- **Catching a cache hit pretending to be speed.** vLLM's prefix caching makes a repeated
  payload look far faster than the real workload, where every frame is new. Sending fresh
  pixels each call is what measures prefill honestly. An early version of this benchmark
  reported prefill as the dominant cost purely because it was re-sending one identical frame.

The effective uplink through the tunnel measured about **500 KB/s**. That is the number
underneath everything else here: 42 KB (one image) rides in roughly a single round trip, while
127 KB (three) costs about 250 ms more.

## The fixes that cost nothing

Two changes make the client faster without changing a single byte the model sees.

**One pooled connection instead of one per step.** `JarvisVLAClient.complete()` called
module-level `requests.post`, which opens a fresh TCP connection every time. Through `ssh -L`
that is not just a handshake — it is a new channel negotiated with the login node before any
frame data moves at all. The client now holds a single `requests.Session` for the life of the
rollout. Measured by interleaving old and new call-by-call so that link drift could not
flatter either side: **426 ms → 364 ms per step**. When the link was slower, the same change
was worth 110 ms.

**Encoding each frame once instead of three times.** History held raw frames, and every
message that referenced a frame re-ran the resize, the JPEG encode and the base64 wrap. With
two frames of history that is three encodes of the same image per step. The client now encodes
once and history holds the resulting data URL, with `encode()` passing strings through
untouched. Worth about 5 ms a step and two large array copies — small next to the network, but
it is pure waste and the code is simpler for its absence.

Together these took a default-settings step from roughly 640 ms to 580 ms, and on a slow link
rather more.

**A compressed tunnel**, which is free in a different sense — it costs nothing in behaviour,
only a flag. The payload is a JPEG, which is incompressible, inflated 33% by base64, which is
entirely compressible. gzip takes a 130 KB request back to 97 KB: it recovers the base64
overhead and nothing else, but that is real. `scripts/connect.sh` now passes `ssh -C`, with
`COMPRESS=0` to opt out on a fast link where the compression CPU would cost more than it saves.

## The fixes that trade something

Everything left in the transport bill is bytes, and the only way to send fewer is to send less
image. Both knobs below are now reachable from the environment — `JARVISVLA_JPEG_QUALITY` was
previously a config field with no way to set it — and both cost behaviour.

| setting | payload | ms/step | steps/s |
|---|---:|---:|---:|
| default (`JARVISVLA_HISTORY=2 JARVISVLA_JPEG_QUALITY=75`) | 127 KB | 369 | 2.7 |
| `JARVISVLA_JPEG_QUALITY=50` | 87 KB | 343 | 2.9 |
| `JARVISVLA_JPEG_QUALITY=40` | 76 KB | 325 | 3.1 |
| `JARVISVLA_HISTORY=0` | 42 KB | 213 | 4.7 |
| `JARVISVLA_HISTORY=0 JARVISVLA_JPEG_QUALITY=50` | 29 KB | 197 | 5.1 |

Quality 75 is upstream's number, so lowering it moves the input off the distribution the model
was evaluated against. Dropping history costs more: the checkpoint was post-trained
conditioning on two previous frames, and without them it has no way to tell "I am mid-swing at
this tree" from "I am looking at a tree". These are settings for iterating quickly, not for
reporting results from.

## The change that actually fixed the demo

At the fastest of those settings a step is 223 ms, which is still under 5 ticks a second, and a
live demo looked it. The obvious suspicion is that the simulator was also slow. It is not.
Measured with no model in the loop at all, one `sim.step` is **26 ms — 38 ticks per second**,
comfortably faster than Minecraft's own 20 Hz. The preview window adds 3 ms and the mp4
recorder 2 ms. Of a 223 ms step, 197 ms was the model and 26 ms was the game.

Nor was there much left to win on the call itself: 95 ms of round trip plus roughly 100 ms of
GPU is 195 ms, and the measurement was 197. Even moving the server onto the same network only
reaches about 110 ms a call. **No amount of squeezing the request makes a synchronous loop
smooth**, because a synchronous loop is asking a 20 Hz game to run at the model's rate.

So the loop stopped being synchronous. With `--async`, the model runs on a worker thread: the
main loop hands it the current frame and immediately carries on stepping the game with the most
recent decision the worker has returned. The game plays at its own speed and the policy
re-decides whenever a reply lands. A frame submitted while a call is in flight replaces any
frame still waiting, so the model always thinks about the freshest observation rather than
working through a backlog it can never catch up with.

Measured in a live Minecraft against the live server, 120 ticks per row:

| mode | ticks/s | calls | ticks per decision |
|---|---:|---:|---:|
| sync (default) | 3.8 | 120 | 1.0 |
| `--async --fps 20` | 19.4 | 26 | 4.6 |
| `--async --fps 0` (uncapped) | 33.8 | 16 | 7.5 |

`--fps` exists because unpaced the sim runs at 34 ticks/s and the result looks fast-forwarded;
20 is Minecraft's own rate. The first tick still costs one full call, since there is nothing to
play until the first reply arrives, and after that the game never stops again.

**What it costs is real.** The policy acts on a frame that is one call old and holds each
decision for about five ticks — a 4 Hz controller driving a 20 Hz game. VPT-style actions are
held keys, which tolerates this far better than a policy emitting discrete one-shot commands
would, but it is not the controller upstream evaluated. Demo with it; measure without it. The
default is unchanged, and the synchronous path behaves exactly as it did before.

## Things found along the way that were not optimizations

Measuring carefully turned up four claims that were simply wrong, three of them in the docs.

`JARVISVLA_CHUNK` was documented as halving round trips by playing several actions per reply.
It does nothing. This checkpoint emits exactly one action bracket per turn — 20 of 20 replies
sampled — so there is never a second action to queue.

The knobs table gave the wrong default for `JARVISVLA_MAX_STEPS` (400; the code says 600), and
the throughput table named its settings without their `JARVISVLA_` prefix, so the rows were not
copy-pasteable. Both fixed.

Finally, the red `can't set up init inventory` that appears at every reset is a false alarm and
can be ignored. minestudio's `InitInventoryCallback` gives the observation `len(items) * 2`
ticks to reflect its `/replaceitem` command — two ticks for a one-item loadout — and the
inventory turns up on the tick after it gives up. Checked directly: `{'stone_axe': 1}` is
present from the first tick onward. The agent has its axe.

## Where it stands

A faithful synchronous rollout at default settings runs at about 2.7 steps/s, up from 1.6. A
demo runs at 19.4 ticks/s, which is Minecraft at its real speed. What remains is the 70% of
each call that is transport, and no client-side change reaches it: the fix is not running the
simulator on the far end of a home uplink. Anything inside Brown's network — the sim on the GPU
box under its own X server, or the client on a login node — removes the upload and the round
trip together and lands near the ~130 ms GPU floor, about 7 steps/s synchronous. That is a port
of the session and environment setup rather than an optimization, and it is the next real step
if the numbers here stop being good enough.

The operational reference — every environment variable, with defaults — lives in
[jarvisvla.md](jarvisvla.md). This document is the reasoning behind those numbers.
