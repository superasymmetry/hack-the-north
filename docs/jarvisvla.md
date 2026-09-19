# JarvisVLA: the sentence-in, keyboard-out controller

[ROCKET-2](rocket2.md) is driven by a point and an interaction type, which says exactly which
object you mean but needs something to produce the point and cannot drive a GUI. JarvisVLA
([arXiv:2503.16365](https://arxiv.org/abs/2503.16365)) is the other end of the trade: a 7B
vision-language model post-trained to emit VPT actions directly, so the goal channel is an
English sentence and nothing else.

```python
from mcagents.agents.jarvisvla import JarvisVLAAgent

agent = JarvisVLAAgent(sim)                       # sim needs the 21-bin camera config
result = agent.run("Chop down the oak log.", stop={"item": "log", "count": 3})
print(result)   # <reason> after N steps (Ns, N steps/s), gained ..., Ns of that waiting on the model
```

Run it: `./scripts/jarvisvla.sh`, or `MC_PORT=9000 ./scripts/jarvisvla.sh` to attach to a
`scripts/mc_server.sh` holder. The plan is a JSON list of instructions; the default lives at
the top of [`mcagents/cli/jarvisvla.py`](../mcagents/cli/jarvisvla.py).

Checkpoint: [`CraftJarvis/JarvisVLA-Qwen2-VL-7B`](https://huggingface.co/CraftJarvis/JarvisVLA-Qwen2-VL-7B),
a fine-tune of Qwen2-VL-7B (~8B params, ~17 GB in bf16).

## It does not run on the laptop

A 7B VLA will not fit on this box's 8 GB next to Minecraft. So the policy runs where the
VLM targeter runs — vLLM on the GPU box — and `mcagents.agents.jarvisvla` is a thin client
over the same tunnel:

```sh
# on the GPU box, under tmux or sbatch
./scripts/serve_jarvisvla.sh

# on the laptop (see the Oscar note below for why this is an IP, not a hostname)
./scripts/connect.sh 172.20.x.x     # = ssh -NC -L 8000:172.20.x.x:8000 szeng26@ssh.ccv.brown.edu
export JARVISVLA_URL=http://127.0.0.1:8000/v1
./scripts/jarvisvla.sh
```

Two checks worth running before a full rollout, in this order:

```sh
python tests/test_jarvisvla_actions.py                    # decoder only: no server, no Minecraft
python -m mcagents.cli.jarvisvla --frame frame.png "Chop down the oak log."
```

The second one is the useful one. If the reply comes back without any
`<|reserved_special_token_...|>` in it, the script says so — that is the failure mode below
that otherwise looks like a bad policy rather than a bad connection.

`ssh -N` printing nothing after "Success. Logging you in..." is the tunnel working, not
hanging — leave it and use a second terminal. Tunnel to the compute node's **IP**, not its
short hostname: the login node cannot resolve those and `-L 8000:gpu4106:8000` dies with
`channel N: open failed: connect failed: Name or service not known`. `hostname -I` on the
node prints two addresses (two networks); `serve_jarvisvla.sh` echoes the first in the `ssh`
line it prints at startup, and that is the one that has worked. `nc -z -v -w3 <ip> 8000` from
the login node says which is routable. The node changes every allocation.

## Three ways this fails silently

Every one of these produces a working HTTP 200 and a useless agent. They are why
`_check_sim()` refuses to start and why `tests/test_jarvisvla_actions.py` exists.

**1. The camera config is part of the checkpoint.** JarvisVLA's action vocabulary ends in two
21-way camera groups — `n_camera_bins=21`, i.e. `CameraConfig(camera_binsize=1,
camera_maxval=10, camera_mu=20)`. `MinecraftSim`'s **default** `CameraConfig` is binsize 2 /
mu 10, which is **11** bins. Under the default every camera token is decoded against the
wrong table and the agent mouses in the wrong units. Nothing in the stack notices.
`JarvisVLAAgent.__init__` raises instead — build the sim with
`**mcagents.agents.jarvisvla.sim_kwargs()` and it is right by construction.

**2. `skip_special_tokens` must be off.** The action *is* a run of special tokens. vLLM
strips those from the response by default, leaving an empty string, which decodes to the null
action — an agent that stands perfectly still while the server reports success. Sent in the
request body by `JarvisVLAClient.complete()`.

**3. The camera bins are hierarchical.** `CameraHierarchicalMapping.to_factored` only reads
the two camera bins when the camera meta-button is set in `buttons`, and the model never
emits that bit. It is reconstructed at decode time from "the bins are not both centred" —
upstream does this, and so does `decode_actions`.

## The action space

An action is a mixed-base number over twelve token groups, bracketed by a begin/end pair:

| group | meaning | base | reserved tokens |
|---|---|---|---|
| 0 | hotbar.1–9 (0 = none) | 10 | 180–189 |
| 1 | none / forward / back | 3 | 190–192 |
| 2 | none / left / right | 3 | 193–195 |
| 3 | none / sprint / sneak | 3 | 196–198 |
| 4–7 | use, drop, attack, jump | 2 each | 199–206 |
| 8 | camera meta-button | 2 | 207–208 |
| 9 | inventory | 2 | 176–177 |
| 10 | camera pitch bin | 21 | 209–229 |
| 11 | camera yaw bin | 21 | 230–250 |

Groups 0–8 multiply out to 8640, VPT's button space; group 9 is not a digit but a flag that
replaces the whole button index with 8640 ("inventory pressed"); groups 10–11 pack as
`pitch * 21 + yaw`. **Index 0 of every group means "not pressed"** — `forward` is token 191,
not 190, which is the kind of off-by-one that costs an afternoon.

`mcagents/agents/jarvisvla_actions.py` decodes the reserved-token *names* out of the reply text, where
upstream re-tokenizes with `AutoTokenizer.from_pretrained(checkpoint)`. Same answer —
`<|reserved_special_token_N|>` is id `151657+N` in this checkpoint's `added_tokens.json` — but
no 7B repo and no transformers call on the laptop. The two decoders were diffed on 400 random
actions and agree exactly, and `null_action_text()` is byte-identical to upstream's
`null_token()`.

## What goes over the wire

```
user       "Chop down the oak log.\nobservation: "   + frame
assistant  "<|...178|><|...204|><|...219|><|...240|><|...179|>"
user       "\nobservation: "                          + frame
assistant  ...
user       "\nobservation: "                          + current frame
```

The instruction rides on the **first** message only; every later turn is bare
`observation:` plus a frame, with the model's own previous replies as the assistant turns.
Before the first reply exists the history is padded with the current frame and the null
action — upstream's cold start, and what the model was evaluated with.

Frames go as JPEG at quality 75, resized to what Qwen2-VL's processor would produce anyway:
640×360 → 644×364, a 13×23 merged grid, 299 image tokens. Quality 75 is upstream's number
(its `processor_wrapper` saves through PIL at the default), so it is the compression the
model was evaluated against, not a bandwidth compromise — but it is also 43 KB a frame and
three frames a step, which is most of the step time over a tunnel, so
`JARVISVLA_JPEG_QUALITY` exists to trade one against the other. Each frame is encoded once
and the data URL is what history holds: with `history_num=2` a frame is sent three times,
and re-encoding it per send was three JPEG passes for one image.

## Knobs

| env var | default | what it does |
|---|---|---|
| `JARVISVLA_URL` | `http://127.0.0.1:8000/v1` | comma-separate for several replicas (round-robin) |
| `JARVISVLA_MODEL` | read from `/v1/models` | only needed if the server serves several models |
| `JARVISVLA_TEMPERATURE` | 0.6 | upstream's rollout value; 0.0 gets stuck holding one key |
| `JARVISVLA_HISTORY` | 2 | frames of history. **0 sends one image per step instead of three — the biggest speed knob, at some cost in behaviour** |
| `JARVISVLA_JPEG_QUALITY` | 75 | upstream's number; lower is fewer bytes on the wire and some distribution shift |
| `JARVISVLA_CHUNK` | 1 | actions to play per call. **A no-op with this checkpoint** — see below |
| `JARVISVLA_ASYNC` | 0 | 1 (or `--async`) thinks on a worker thread so the game never waits. **For demos, not for measurements** — see below |
| `JARVISVLA_FPS` | 20 | ticks/s to hold the game to; 0 uncaps. Only reachable in async — a synchronous rollout is far below it |
| `JARVISVLA_MAX_STEPS` | 600 | per-instruction ceiling |
| `CITY` | 0 | put the agent in a city (on by default for ROCKET-2, not here; `--city` also works). Loads `worlds/city.zip` if `scripts/bake_city.sh` has built one, else builds it at every reset |
| `JARVISVLA_PREVIEW` | 1 | live window of the frame the model is looking at; 0 (or `--no-preview`) is off |

Everything in that table is also a flag: `--max-steps`, `--log-every`, `--no-record`,
`--no-preview`, `--city`, `--plan`. `stop` takes the same vocabulary ROCKET-2 does — an int step budget, `{"steps": n}`,
`{"item": "log", "count": 3}` (plus `"mode": "total"`), `{"stat": "mine_block", "match":
"log", "count": 3}`, or any `callable(agent) -> bool` — so a planner that can drive one
controller can drive the other. Item names are matched loosely, so `"log"` covers
`oak_log`/`birch_log`/….

## Throughput: measured

The knobs and the numbers are here; [performance.md](performance.md) is the reasoning behind
them — how the step was decomposed, what each fix cost, and what is left.

Unlike the targeter, which is called once per *goal*, this is called once per *step*: a
rollout is a strictly serial chain of round trips, each carrying three JPEGs up and a handful
of tokens back. Measured 2026-09-04 from the laptop, through the standard tunnel to a
compute-node vLLM, replaying real consecutive frames from a recorded episode — **the tunnel
is the limit and the GPU is idle ~80% of the time**:

| where a step goes | ms | share |
|---|---:|---:|
| round-trip floor (tunnel + HTTP, 1 token in, 1 out) | 95 | 15% |
| uploading 127 KB of base64 JPEG | 350 | 55% |
| a fresh TCP connection — a new `ssh -L` channel — per call | 110 | 17% |
| GPU prefill + decode (968 prompt tokens, 8 generated) | 130 | 20% |
| **total** | **~640** | 1.6 steps/s |

To price a part on its own: a POST with `"model": "nope"` is read in full and rejected before
any model work, which weighs the bytes; `max_tokens=1` against a real prompt separates prefill
from decode; and repeating one payload versus sending fresh pixels separates a prefix-cache
hit from real prefill. Effective uplink through the tunnel measured ~500 KB/s, so 42 KB (one
image) rides in about one RTT while 127 KB (three) costs ~250 ms more.

The last two rows of that table are what the client can do something about, and it now does:
`JarvisVLAClient` holds one pooled `requests.Session` for the rollout instead of opening a
connection per step, and each frame is JPEG-encoded once and kept as its data URL rather than
re-encoded for each of the three messages it appears in. Interleaved A/B on a live server:
429 → 365 ms a step, and up to ~110 ms when the link is slower and connection setup costs more.

What is left is bytes, and only two things move them:

| setting | payload | ms/step | steps/s |
|---|---:|---:|---:|
| default (`JARVISVLA_HISTORY=2 JARVISVLA_JPEG_QUALITY=75`) | 127 KB | 369 | 2.7 |
| `JARVISVLA_JPEG_QUALITY=50` | 87 KB | 343 | 2.9 |
| `JARVISVLA_JPEG_QUALITY=40` | 76 KB | 325 | 3.1 |
| `JARVISVLA_HISTORY=0` | 42 KB | 213 | 4.7 |
| `JARVISVLA_HISTORY=0 JARVISVLA_JPEG_QUALITY=50` | 29 KB | 197 | 5.1 |

Both cost behaviour, and history costs more of it than quality does — the checkpoint was
post-trained with two frames of it. Treat the bottom rows as a way to iterate quickly, not as
the configuration to report results from. `scripts/connect.sh` passes `ssh -C` for the same
reason: the payload is a JPEG (incompressible) inflated 33% by base64 (compressible), so gzip
takes 130 KB back to 97 KB and recovers exactly that overhead.

Two things that look like knobs and are not. `JARVISVLA_CHUNK=2+` does nothing with this
checkpoint: it emits exactly one `<|reserved_special_token_178|>…179` bracket per turn (20 of
20 replies sampled), so there is never a second action to queue. And extra replicas do not
speed up one rollout — one call is in flight at a time. They let several rollouts run at once:
launch the serve script again on another GPU and port (`PORT=8001 GPUS=1
./scripts/serve_jarvisvla.sh`) and comma-separate `JARVISVLA_URL`.

The real fix for the 70% that is transport is not a knob: it is not running the sim on the far
end of a home uplink. Anything inside Brown's network — the sim on the GPU box under its own
X server, or the client on a login node — deletes the upload and the RTT together and lands
near the ~130 ms GPU floor, about 7 steps/s.

## Demoing it: don't make the game wait

The simulator is not what is slow. Measured on this laptop with no model in the loop, one
`sim.step` is **26 ms — 38 ticks/s**, comfortably above Minecraft's own 20 Hz; the preview
window adds 3 ms and `RecordCallback` 2 ms. So of a 223 ms step at the fastest settings above,
197 ms is the model and 26 ms is the game, and a synchronous rollout looks like a quarter-speed
Minecraft because that is exactly what it is.

`--async` (or `JARVISVLA_ASYNC=1`) resolves the rate mismatch the other way round. The model
runs on a worker thread: the loop hands it the current frame and keeps stepping the game with
the newest decision it has returned, so the game plays at 20 ticks/s and the policy re-decides
whenever a reply lands. Measured in a live Minecraft, 120 ticks per row:

| mode | ticks/s | calls | ticks per decision |
|---|---:|---:|---:|
| sync (default) | 3.8 | 120 | 1.0 |
| `--async --fps 20` | 19.4 | 26 | 4.6 |
| `--async --fps 0` (uncapped) | 33.8 | 16 | 7.5 |

`--fps` exists because unpaced the sim runs at 34 ticks/s and the demo looks fast-forwarded;
20 is Minecraft's own rate. The first tick still costs one full call — there is nothing to
play until the first reply lands — and after that the game does not stop again.

What this costs is real, and it is not a free lunch: the policy acts on a frame that is one
call old and holds each decision for ~5 ticks, so it is a 4 Hz controller driving a 20 Hz
game. VPT-style actions are held keys, which tolerates that better than a policy emitting
discrete one-shot commands would, but **it is a different controller from the one upstream
evaluated**. Demo with it; measure without it.

A note on a red herring you will see either way: `can't set up init inventory` in red at every
reset. The item is there — minestudio's `InitInventoryCallback` gives the observation
`len(items) * 2` ticks to reflect the `/replaceitem`, which for a one-item loadout is two, and
the inventory turns up on the tick after it gives up. Verified: `{'stone_axe': 1}` from the
first tick onward.

`TaskResult` reports `calls` and `wait` (seconds blocked on the server, and the ms/call that
works out to) next to `steps` and `seconds`, which is what says whether a slow run is the
network or the game.

## Prompts

Stay on the distribution the model was post-trained on: short, imperative, one primitive
each. Its own `assets/instructions.json` is the reference, and the phrasings look like
`"Mine the oak log."`, `"Chop down the oak log."`, `"Hunt a sheep."`, `"Break the stone
block."`, `"Craft a crafting table."`. Compositional goals degrade badly — sequence them
from a planner instead.

Crafting is the one job ROCKET-2 cannot do at all: JarvisVLA drives the inventory and
crafting-table GUI itself, which is the gap [rocket2.md](rocket2.md) lists under "Crafting
menus". That path is also the least exercised here — upstream runs a scripted `CraftWorker`
to open the crafting table *before* handing control to the model, and this repo does not port
that helper. A bare `"Craft a crafting table."` asks the model to open the
inventory itself.

## Where the code comes from

Nothing is vendored. `mcagents/agents/jarvisvla*.py` reimplements the two pieces of
[CraftJarvis/JarvisVLA](https://github.com/CraftJarvis/JarvisVLA) a client actually needs —
the prompt shape from `evaluate/agent_wrapper.py` and the action codec from
`inference/action_mapping.py` — against the same `minestudio` this repo already has. Its
`pip install -e .` pulls deepspeed, ray, trl and vllm, none of which belong in the env that
runs Minecraft, and its `evaluate.py` calls `InitInventoryCallback` with keyword arguments
this minestudio version does not have.
