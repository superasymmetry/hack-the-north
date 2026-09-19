# The capture side

[mcagents/local_client.py](../mcagents/local_client.py) is the half that runs on the laptop:
microphone and game view in, transcript and frames out, over **one** WebSocket to the
interaction model on the GPU box. The model itself is remote and already reachable through a
Cloudflare tunnel; nothing in this file knows what it does with the words.

```bash
conda activate ./.conda-env
pip install websockets                     # the only new dependency

cp .env.example .env && chmod 600 .env     # once: then edit the two AGENT_ lines
./scripts/local_client.sh --check          # is the tunnel there? one attempt, a verdict
./scripts/local_client.sh --text           # prove the socket and the token first
./scripts/local_client.sh                  # then the microphone
./scripts/local_client.sh --list-devices   # which input is actually live
python tests/test_local_client.py          # no microphone, no tunnel, no GPU
```

## Running the whole loop

Three terminals. `scripts/env.sh` loads `.env` for every wrapper, so both processes pick up
the same tunnels and the same two channel paths without exporting anything by hand.

```bash
# 1. hold one Minecraft open, so restarting the agent does not cost a boot
bash scripts/mc_server.sh                       # port 9000

# 2. ROCKET-2, attached to it, waiting to be told what to do
MC_PORT=9000 bash scripts/rocket2.sh --listen

# 3. the voice client
./scripts/local_client.sh --check               # prove the tunnel first
./scripts/local_client.sh
```

Start ROCKET-2 first and the first thing you say already has a picture attached. When both
are up, the client says so:

```
frames: /tmp/mcagents-frame.jpg is live, sending one every 3s (11.4 KB)
connected to wss://your-tunnel.trycloudflare.com
connected to wss://your-goal-tunnel.trycloudflare.com (goals)
```

and a turn looks like this, in the client:

```
  sent: mine the oak tree   (asr 141 ms . end->sent 156 ms)
[12:03:41] agent: On it -- heading for the tree by the fountain.
goal: Mine 'the oak tree' stop={'item': 'oak_log', 'count': 3}
```

and in the ROCKET-2 terminal:

```
[listen] Mine at (412.0, 173.0)
[rocket2] Mine@(412,173) stop={'item': 'oak_log', 'count': 3} mask=18204px
[rocket2] item after 143 steps (11.4s), gained oak_log+3, visibility 0.87
[listen] waiting for goals in /tmp/mcagents-goals
```

`--listen` given on its own waits rather than running the default plan; given alongside
`--plan` or `--goal` it picks up after those finish. Between goals the world keeps
stepping — a frozen sim stops publishing frames within seconds, and the planner is then
choosing what to do next from a picture of a world that has stopped.

The two GPU tenants coexist: ROCKET-2 and SAM-2 hold the card, and the recogniser here is
CPU-only for [its own reasons](voice.md), so nothing contends.

## System packages

Neither of the two you would expect to need is a problem here.

**PortAudio** is required — `sounddevice` is a binding, not a library — and it is already
installed in `.conda-env` (`conda install -c conda-forge portaudio`, which
[docs/voice.md](voice.md) covers). Do *not* `apt install portaudio19-dev` to fix an import
error: `sounddevice` loads the conda one, and a second copy on the system just makes the
question of which is live harder to answer.

**ffmpeg is not needed.** It gets pulled into ASR instructions everywhere because Whisper's
reference implementation shells out to it to decode media files. Nothing here decodes a
file — audio arrives from `sounddevice` as float arrays and goes straight into the model.

## Why this does not use faster-whisper

It was the obvious first choice, so it is worth writing down why it is not the one. Measured
on this laptop — Core Ultra 9 285H, 16 cores, CPU int8, `beam_size=1`, median of five:

| model | 1.0s utterance | 2.0s utterance | 5.0s utterance |
|---|---:|---:|---:|
| `tiny.en` | — | 283 ms | — |
| `base.en` | 494 ms | 514 ms | 494 ms |

The number to look at is not the size of it but the *flatness* of it. A five-second phrase
costs what a one-second phrase costs, because Whisper's encoder zero-pads every clip to
thirty seconds. That is a floor, not a curve: it does not come down with more threads, a
smaller beam, or shorter utterances, because none of those change how much mel the encoder
is handed.

Against a budget of 100–150 ms for finalisation inside a ~500 ms speech-to-first-audio-back
target, `base.en` spends the entire end-to-end budget by itself before the VLM is even given
the sentence, and `tiny.en` spends most of it while giving up the accuracy that made the
sentence worth sending. So the recognition here is [voice.py](../mcagents/voice.py)'s
`Listener` — Moonshine v2 `SMALL_STREAMING`, 123M parameters, ~148 ms at 7.84% WER — whose
streaming encoder does the work *while the person is still talking*, so what is left at
end-of-speech is the tail rather than the whole job. Reusing it also inherits the one thing
on this machine that is expensive to rediscover: which microphone is real.

The client now defaults to `TINY_STREAMING` anyway, with a 0.3 s VAD window. With Minecraft
and ROCKET-2 on the same laptop, `SMALL_STREAMING` finalised in 500–900 ms and overflowed the
audio input, which cost far more than its accuracy bought. `VOICE_ARCH=SMALL_STREAMING`
brings it back when the sim is running somewhere else.

There is no `webrtcvad` here either. Moonshine decides where a phrase ends from the audio,
inside the library. A separate VAD would be a second opinion about a boundary the recogniser
has already committed to, arriving after the fact.

## The wire

One connection for the whole session. A TLS handshake per transcript delta would cost more
than the model's entire turn, which is the entire reason the socket is long-lived rather
than per-utterance.

```json
{"text": "mine the diamond ore", "final": true}
{"kind": "frame", "image": "<base64 jpeg, no data: prefix>"}
```

Partials go too, as `final: false`. The server enqueues an utterance only on `final: true`
and ignores the rest, so today they cost nothing — and endpointing and barge-in will both
hang off them later.

`Outbox` is where the kinds stop being interchangeable, and it is the only real idea in
the file:

- **A partial is worthless once a newer one exists.** An unsent partial is *replaced*, never
  queued behind. Under any backpressure the right partial to send is the current one, and a
  backlog of stale ones is strictly worse than nothing.
- **A frame is a partial in pixels.** One slot, newest wins, dropped on a reconnect. A
  picture of the game three seconds ago is not worth the bytes it would cost to deliver.
- **A final is the utterance.** Losing it means the person says it again. So finals queue,
  and they survive a reconnect — up to `max_final_age` (30s), after which a Minecraft
  command is no longer the command anyone meant and is dropped and counted.

The order out is finals, then the frame, then the partial. Finals first because they are the
command. The frame ahead of the partial because the server ignores partials and does not
ignore frames — and because a frame arrives every three seconds while a partial regenerates
three times a second, so this cannot starve partials where the other order could sit on a
frame for the length of a long sentence.

Two failures around that are easy to write and impossible to see:

1. Taking a final out of the outbox and *then* failing to send it loses the utterance at
   exactly the moment the reconnect logic exists to survive. It is held in `inflight`,
   outside the connection's scope, until a `send()` actually returns.
2. Waiting only on the outbox means that with nothing to say, a dead socket goes unnoticed
   until the person next speaks — so every reconnect eats an utterance and the backoff is
   paid mid-sentence. `next_to_send()` watches the outbox and the connection together.

Both are covered in [tests/test_local_client.py](../tests/test_local_client.py), which is
also where the stub server lives.

A successful `send()` means the frame reached the kernel, not that the server handled it.
Closing that last gap needs an ack in the protocol, and the protocol has none.

## Testing the tunnel

```
$ ./scripts/local_client.sh --check
  url      wss://polite-otter-cascade.trycloudflare.com
  token    32 chars
  dns      polite-otter-cascade.trycloudflare.com -> 104.21.9.11, 172.67.144.2  (14 ms)
  upgrade  101, connected            (312 ms)
  send     {"text": "connection check", "final": false}  (1 ms)

  ok -- tunnel, token and protocol all good. Nothing was enqueued: the probe
  was final=false, which the server accepts and ignores.
```

`--check` exists because the reconnect loop is right for a session and wrong for a question.
With the tunnel down, `--text` retries politely forever and never tells you what is wrong;
this connects **once**, times each stage, and exits. It sends only a partial, so it exercises
the whole path — DNS, TLS, the upgrade, the header, a frame — without putting words in the
agent's mouth.

The point of naming each stage is that the three failures which get mistaken for each other
do not look alike:

| what you see | what it is | exit |
|---|---|---|
| `dns cannot resolve ...` | the hostname is wrong, or `cloudflared` is not running. A quick tunnel gets a new hostname every restart | 1 |
| `upgrade failed -- ConnectionRefused` / timeout | the hostname is real but the tunnel process has since stopped | 1 |
| `upgrade HTTP 502` | the tunnel is up, the Slurm job behind it is not | 1 |
| `upgrade HTTP 401/403` | the token, refused at the handshake | 2 |
| `upgrade 101` then `closed 1008` | the token — and the tunnel and server are both fine | 2 |

That last row is the one the protocol actually specifies, and it is the one that reads like
a network fault if you are not looking: the connection *succeeds*, and the refusal arrives a
moment later out of the first `send()`.

Then, in order: `--text` to prove the link end to end with the audio stack out of the
picture, and only then the microphone.

## Dropping and being refused are not the same thing

The remote runs under Slurm and the job can die mid-session, so a dropped socket is normal
operating procedure: reconnect with exponential backoff (0.5s doubling to 30s, jittered so a
client and a restarting server do not fall into lockstep) and keep the microphone open
throughout. Nothing needs restarting.

A **refused** socket is the opposite. A wrong token is closed with 1008, and retrying it
fails identically every few seconds while looking exactly like a flaky network. So 1008 —
and a 401/403 on the upgrade request — stops the client and says which of the token and the
URL to go and look at. Every other status, a 502 among them, is the tunnel or the job being
absent, and is retried.

## The latency it prints

```
  sent: mine the diamond ore   (asr 141 ms . end->sent 156 ms)
```

`asr` is Moonshine's own reported cost for the final transcription pass. `end->sent` is the
one the budget is about: from the end of the speech to the bytes leaving this machine.

End-of-speech is taken from the phrase's own place on the audio clock —
`TranscriptLine.start_time + duration`, anchored to a monotonic reading taken as the stream
opens. Timing from the callback instead would measure how promptly we reacted to being told
the phrase was over, which is a small, flattering number that answers a different question.
The sound card's clock is not the system's, so the result is sanity-checked before it is
believed: an anchor implying speech ended in the future, or more than 30s ago, reports `asr`
alone rather than a confident wrong number. `--timing` prints the raw components.

In `--text` mode there is no speech, so the line reads `queued->sent` and measures the
socket path alone.

## Sending the game view

A spoken instruction is half a message. "Mine *that*" asks the model to resolve a pointing
word against a world it cannot see, so every three seconds the agent's own view goes up
alongside the words.

The pixels are not this process's to take. The Minecraft window belongs to the ROCKET-2 run,
which is a **separate process on this same laptop** — it has the sim, and this file has the
socket, and neither has both. [mcagents/frames.py](../mcagents/frames.py) is the seam:

```
rocket2 process                        local_client process
  sim.step()                             every 3s: read_latest()
  obs["image"]  (224x224 RGB)                     |
      |  FramePublisher.offer(), 1s             base64
      v                                           v
   /tmp/mcagents-frame.jpg  <--------------  one socket -> GPU box
```

A **file**, not a socket or a queue, because of what the data is. A frame is worthless the
moment a newer one exists, so the channel wants exactly one slot, last writer wins, no
backlog and no reader to block on — which is a file and is not any queue. It also means
neither process has to exist for the other to start: the client polls a path that may never
appear, the agent writes to one nobody is reading, and restarting either is not an event.
Running the voice client with no agent up sends nothing and says nothing.

`os.replace` is what makes it safe. It is atomic within a filesystem, so a reader gets the
whole previous JPEG or the whole new one and never a half-written frame; writing in place
would hand out torn files a few times a minute, and they would decode as garbage rather than
as an error.

Three things the client refuses to send, each for its own reason:

| | why |
|---|---|
| a frame older than `frame_max_age` (10s) | the file outlives the process that wrote it. A crashed or ctrl-c'd agent leaves a perfectly valid JPEG lying there forever, and sending it narrates a session that has ended |
| the same frame twice | mtime says whether the publisher has moved. A paused or wedged agent goes quiet rather than repeating itself down the tunnel every three seconds |
| a file over 4 MB, or an empty one | `MCAGENTS_FRAME_PATH` is a well-known path and a typo can point it at anything. A real frame here is ~12 KB, so this is checked from the open handle *before* the read rather than after it |
| any frame, when `--no-frames` | words only |

**Why `obs["image"]` and not `info["pov"]`.** `obs["image"]` is the 224×224 the policy is
actually looking at, so what the remote model is shown is what drove the last action.
`info["pov"]` is the same view at 640×360 and is one line away in `Rocket2Agent._after_step`
if the model ever needs the detail more than it needs the correspondence.

**What it costs.** Measured over 44 frames of a real run
(`logs/jarvisvla/episode_1.mp4`, downsized to what the policy sees), JPEG q80 gives
10.3–13.7 KB, median **11.5 KB** — about 15 KB of base64 on the wire, or **5 KB/s** at one
frame every three seconds. The encode is far below one 20 Hz step, which is what lets the
publisher sit in the step loop at all.

**Why the two intervals differ.** The publisher writes every 1s and the client sends every
3s. They poll independently, so publishing at the same cadence the client reads at would
make the frame it picks up as much as two intervals old; at 1s against 3s, what goes on the
wire is never more than about a second stale.

## The way back

Two tunnels, because the server has two things to say and they are not the same kind of
thing. One is the conversation with the person; the other is the JSON the agent runs.

```
                 AGENT_WS_URL          text + frames  -->
   local_client  <---------------->    <-- what the model says back
                 AGENT_GOAL_WS_URL     <-- {"point": ..., "interaction": ..., "stop": ...}
                        |
                   /tmp/mcagents-goals/   (one file per goal, oldest first)
                        |
                 rocket2 --listen  ->  set_goal() -> the world moves -> new frames go up
```

The uplink became **bidirectional**: `run_link` now sends and receives at once. It has to be
at once — a receive loop that took its turn between sends would only notice the model's
answer when the person next spoke, which is exactly backwards, since the answer is what they
are waiting for. Replies are `{"text": ...}` and print as `[12:03:41] agent: ...`. When a
goal ends the agent also speaks unprompted, as `{"text": ..., "reason": "arrived"}`, which
prints as `agent (done): ...` (green on a terminal) -- or `agent (lost): ...`, or
`agent (<reason>): ...` for anything newer. Both kinds are spoken. Anything that is not JSON
with a string `text` is ignored.

The goal tunnel connects separately. Goals come down it; `goal.status` messages go back up
— what the agent left in `/tmp/mcagents-status` — so the server can log why each goal ended
instead of guessing from frame motion:

```json
{"kind": "goal.status", "event": "ended", "reason": "arrived", "box": [0.18, 0.0, 0.82, 0.74],
 "t": 1789354931.805, "interaction": "Approach", "id": "b", "steps": 135, "seconds": 10.23}
```

`event` is `started` / `updated` / `ended` / `rejected`; `box` is frame fractions or null;
`t` is this laptop's epoch clock; `id` is echoed when the goal had one. While an Approach
runs the client looks for frames every `AGENT_APPROACH_FRAME_INTERVAL` (0.5 s), which puts
the agent's once-a-second publish on the wire at ~1 fps; otherwise every
`AGENT_FRAME_INTERVAL`. Both tunnels go through `hold_open`, so they reconnect, back off and
refuse a bad token identically; a 1008 on either stops the client and names the token.

### Goals queue; frames do not

[mcagents/goals.py](../mcagents/goals.py) is the mirror of `frames.py` and deliberately the
opposite shape. A frame is worthless once a newer one exists — one slot, last writer wins. A
goal is a **command**: somebody said it and is waiting to see it happen, so goals queue, come
back oldest first, and none is overwritten by the next. It is the same distinction `Outbox`
makes between a partial and a final, one layer down.

One file per *message*, named by nanosecond, so the queue needs no locking and the order is
in the names; several goals in one message are queued as one `{"goals": [...]}` entry, so
the agent can tell a plan from a correction. A goal that has waited more than 60s is dropped
rather than run late — and logged, and reported upstream as `rejected` / `stale`.

Queueing is the channel's shape, not the agent's policy: `rocket2 --listen` reads the queue
before every step, so a new message preempts, merges into or cancels the goal in progress —
see [docs/rocket2.md](rocket2.md#taking-goals-from-a-person-live). To halt:

```json
{"cancel": true}
```

### What comes down the goal tunnel

A **plan entry** — the same JSON `--plan` files hold, which
[mcagents/cli/plan.py](../mcagents/cli/plan.py) already documented. That is the whole reason
this is a thin channel and not a protocol: the goal vocabulary existed, a planner was always
meant to emit it, and the server is a planner that happens to be listening to a person.

```json
{"point": "the oak tree", "interaction": "Mine", "stop": {"item": "oak_log", "count": 3}}
```

One goal, a list of them, or a list wrapped in `{"goals": [...]}` are all read, in order. A
`"kind"` wrapper key is stripped. `point` may be pixels, `[u, v]` fractions with
`"normalized": true`, or a **description** for the targeter to find in the current frame —
which is the one that lets the server write a whole goal without seeing coordinates.

An **in-place turn** is the one message that is not a plan entry, because it has nothing to
point at:

```json
{"turn": 90}
```

Integer-or-float yaw degrees, `0 < |turn| <= 180`, positive right (clockwise from above),
negative left; pitch is left alone. It travels alone: `{"turn": 0}`, `{"turn": 270}`, a
non-number, a turn carrying any goal key, or a turn inside a list of goals is refused whole
-- printed as `turn: rejected ...` and sent up as `rejected` / `invalid` -- rather than
dropped. (Before this, a turn had no goal key and vanished without a line.) ROCKET-2 cancels
whatever is running, rotates by camera actions alone -- no movement, no attack or use, no
policy, no targeter -- at most `--turn-step` (15) degrees a step, then stands still. When the
rotation is done it reports
`{"kind": "goal.status", "event": "ended", "reason": "turned", "turn": 90, "steps": 6, ...}`.
See [docs/rocket2.md](rocket2.md#taking-goals-from-a-person-live).

Two refusals worth knowing. An object with no goal key in it is **not** made into a goal: a
stray `{"status": "ok"}` turned into one sends the agent at nothing, and silently. And a
goal ROCKET-2 cannot take — a bad `stop`, an unknown interaction — is skipped with a
reason, leaving the agent free for the next one.

## Where it goes wrong

| symptom | cause |
|---|---|
| `the tunnel host and token are still the placeholders` | edit `.env` (or pass `--url` / `--token`) |
| `rejected by the server: ... 1008` | the token. Deliberately not retried — see above |
| `disconnected -- HTTP 502; retrying in 1.0s` | the tunnel is up, the Slurm job behind it is not |
| **you speak and nothing appears** | the wrong input device. PortAudio cannot see PipeWire; `--list-devices` measures each one, and the chosen one is printed at startup |
| the first run hangs for ~45s | Moonshine downloading weights, once, to `~/.cache/moonshine_voice` |
| **no `frames: ... is live` line** | nothing is publishing. ROCKET-2 is not running, or is running with `MCAGENTS_FRAME_INTERVAL=0`, or the two processes disagree about `MCAGENTS_FRAME_PATH` |
| `frames: nothing published for 10s` | the agent stopped, or its step loop has stalled. The client goes quiet rather than resending the last frame |
| `frames: publishing to ... failed` | the *agent* side, once, then it gives up on the channel — a broken side channel must not take a rollout down |
| `goals: AGENT_GOAL_WS_URL is not set` | the model can talk back but nothing will drive ROCKET-2. Set the second tunnel |
| goals print in the client, nothing happens in the game | ROCKET-2 is not running with `--listen`, or the two disagree about `MCAGENTS_GOAL_DIR` |
| `[listen] skipped -- nothing in view matches ...` | the targeter could not find what the server described in the current frame. `--targeter vlm` reads descriptions far better than OWLv2 |
| `[listen] skipped -- unknown stop keys [...]` | the server sent a `stop` outside the vocabulary in `mcagents.agents.base`. The agent is unharmed and takes the next goal |
| `turn: rejected {...} -- ...` | the server sent a turn outside `0 < \|turn\| <= 180`, or with goal keys beside it. Nothing reaches ROCKET-2; the reason goes up as `rejected` / `invalid` |
| `OSError: PortAudio library not found` | `conda install -c conda-forge portaudio` |

## Configuration

Everything is an environment variable, and `scripts/env.sh` loads `.env` from the repo root
before running anything, so none of it has to be exported by hand. `.env` is gitignored
because the token is in it; [`.env.example`](../.env.example) is the tracked template and
documents the rest of the repo's variables too.

Two properties of that loader are deliberate. A variable already set in your shell **wins**
over the file, so a one-off `AGENT_WS_URL=wss://other ./scripts/local_client.sh` still works
— useful, since a quick tunnel gets a new hostname every restart. And the file is *parsed*,
not sourced: it holds a token, and `source` on a secrets file executes whatever is in it.
Trailing whitespace on an unquoted value is stripped, because a token with a trailing space
is otherwise a 1008 you will spend an evening on; quote the value to keep it byte-for-byte.

| variable | default | |
|---|---|---|
| `AGENT_WS_URL` | the placeholder | the tunnel, `wss://...` |
| `AGENT_TOKEN` | the placeholder | sent as `X-Agent-Token` |
| `AGENT_UPDATE_INTERVAL` | `0.3` | seconds between partials — not phrase length |
| `AGENT_FRAMES` | `1` | `0` sends words only (same as `--no-frames`) |
| `AGENT_FRAME_INTERVAL` | `3.0` | seconds between frames on the wire |
| `MCAGENTS_FRAME_PATH` | `/tmp/mcagents-frame.jpg` | the seam; **both** processes read it |
| `MCAGENTS_FRAME_INTERVAL` | `1.0` | the *agent* side: seconds between publishes, `0` disables |
| `AGENT_GOAL_WS_URL` | unset | the second tunnel: goals for ROCKET-2. Unset means no agent is driven |
| `MCAGENTS_GOAL_DIR` | `/tmp/mcagents-goals` | the goal queue; **both** processes read it |
| `MCAGENTS_STATUS_DIR` | `/tmp/mcagents-status` | goal statuses from ROCKET-2, sent up the goal tunnel; **both** processes read it |
| `AGENT_APPROACH_FRAME_INTERVAL` | `0.5` | seconds between frame checks while an Approach runs; `0` keeps `AGENT_FRAME_INTERVAL` |
| `VOICE_ARCH` | `TINY_STREAMING` | `SMALL_` / `MEDIUM_STREAMING` are more accurate and heavier |
| `VOICE_VAD_WINDOW` | `0.3` | seconds of silence that end a phrase; Moonshine's own is 0.5 |
| `AGENT_JOIN_WINDOW` | `0.4` | seconds a finished phrase waits to be joined by the next, extended while the mic hears speech; `0` sends each phrase at once (`--join-window`) |
| `MOONSHINE_ORT_SINGLE_THREAD` | `1` | set by voice.py; `0` restores ONNX Runtime's thread-per-core pool, which spins every core |
| `AGENT_SPEAK` | `1` | `0` prints replies without saying them (`--no-speak`) |
| `VOICE_TTS_VOICE` | `piper_en_US-lessac-medium` | any id from `list_tts_voices("en_us")`; `kokoro_*` is nicer and ~4x the CPU |
| `VOICE_MUTE_WHILE_SPEAKING` | `1` | `0` keeps the mic open during a reply -- headphones only (`--no-mute`) |
| `AGENT_CUE` | `assets/let-me-do-it-for-you.mp3` | played via `pw-play` when a new goal arrives; empty or `--no-cue` turns it off |
| `VOICE_DEVICE` | PipeWire's source, else probed | index or name substring |
| `VOICE_AUDIO_LOG` | `0` | `1` to let PortAudio's probe chatter through |

`--no-partial` sends and shows finals only; `--no-keyterms` drops the Minecraft vocabulary
bias that `voice.py` applies to the decoder. `--frame-interval` and `--frame-path` stand in
for their environment variables, and `--timing` prints a line per frame with its
`published->sent` age (normally only the transitions in and out of sending are reported —
a frame every three seconds would be three lines a minute on a display whose job is to show
a sentence being recognised).
