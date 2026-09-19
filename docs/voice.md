# Talking to it

[mcagents/voice.py](../mcagents/voice.py) turns a microphone into the English that
[JarvisVLA](jarvisvla.md) already takes as its goal channel, and streams a reply back. It is
one file, it imports neither minestudio nor torch, and the recognition never touches the GPU
— which is the whole reason it can run while a rollout has the card.

```bash
conda install -c conda-forge portaudio     # once: sounddevice is a binding, not a library
pip install moonshine-voice

./scripts/voice.sh --no-llm                # transcription only, nothing to connect to
./scripts/voice.sh                         # with a chat server on VOICE_LLM_URL
./scripts/voice.sh --say "hello"           # skip the microphone, test the LLM leg alone
./scripts/voice.sh --no-partial            # only finished phrases, no live text
./scripts/voice.sh --list-devices          # which input is actually live
python tests/test_voice.py                 # no microphone, no server, no GPU
```

## Why Moonshine and not Whisper

Whisper's encoder is fixed at 30 seconds and zero-pads whatever you hand it, so a
two-second phrase costs the same compute as a half-minute one. That is a constant floor
under every utterance, and it lands on the one number a conversation is judged by.

Moonshine v2 ([arXiv:2602.12241](https://arxiv.org/abs/2602.12241)) processes the audio's
actual length, and its streaming encoder caches encoder and decoder state *while the person
is still talking* — so most of the work is done by the time they stop. Published latencies
on an M3, against the Whisper tier each matches:

| arch | params | latency | WER | CPU, as a share of the audio's own duration |
|---|---:|---:|---:|---:|
| `TINY_STREAMING` | 34M | 50 ms | 12.01% | 8% |
| **`SMALL_STREAMING`** (default) | 123M | 148 ms | 7.84% | 18% |
| `MEDIUM_STREAMING` | 245M | 258 ms | 6.65% | 29% |

Small is the default because 12% WER is too rough for open conversation you intend to hand
to an LLM, while 18% of one core is nothing next to what Minecraft is already doing. Tiny is
the right pick if the vocabulary is a closed command grammar rather than speech.

The whole install is a 20 MB wheel plus numpy and sounddevice — no torch, no CTranslate2,
no onnxruntime of your own. Weights are ~138 MB, downloaded once to
`~/.cache/moonshine_voice` and reused.

## Which microphone

The single most expensive failure here, because it is completely silent: you speak, and
nothing appears.

conda-forge's PortAudio is built with **only the ALSA host API**. It cannot see PulseAudio
or PipeWire, so "the default input" means the first ALSA capture device on the lowest card
— on this laptop `hw:1,0`, the analog headset jack, silent with nothing plugged into it.
The built-in microphone array is `hw:1,6`, which is what PipeWire had selected all along and
what PortAudio knew nothing about. Measured with nobody speaking:

```
   [4] sof-hda-dsp: - (hw:1,0)   peak 0.0002   silent -- nothing plugged in?
 * [8] sof-hda-dsp: - (hw:1,6)   peak 0.0192   live
```

So `choose_input()` does not accept PortAudio's answer. It asks PipeWire what it is actually
using (`wpctl inspect @DEFAULT_AUDIO_SOURCE@` gives `alsa.card` and `alsa.device`, which
name a `hw:C,D` that PortAudio *does* list), and failing that listens to each device for a
moment and takes the one with signal — but only when the margin is decisive, because a close
call is a guess and a wrong guess here is invisible. The device it picked is printed at
startup, every time. `--device N` or `VOICE_DEVICE` overrides all of it.

```bash
./scripts/voice.sh --list-devices          # every input, with its measured level
```

One detail worth keeping: the probe listens for 1.2s and throws the first 0.5s away. A
digital microphone array takes about that long to come up, and over a 0.3s window the live
mic here reads 0.0004 — indistinguishable from the dead jack. A probe short enough to feel
free is a probe that confidently reports the wrong device.

## Watching it recognise

In-progress text is printed on a line that rewrites itself, so a phrase visibly firms up
while you are still talking — `mine the die` becoming `mine the diamond ore` — and then
settles into a `you:` line with the recognition latency beside it. It is the streaming
encoder made visible, and it is also the most direct way to see what the LLM is about to be
handed. `--no-partial` turns it off.

`LiveLine` does the drawing, and two of its details are load-bearing: a revision that is
*shorter* than what it replaces has to paint over the tail of the old one, or the difference
stays on screen as debris; and partials arrive on Moonshine's worker thread, so the line is
held back with `pause()` while a reply is printing rather than redrawing through it. Both
are covered in [tests/test_voice.py](../tests/test_voice.py) — they fail silently otherwise,
and only on a real terminal.

## The shape of it

Two halves that do not know about each other, so either can be tested alone:

- **`Listener`** wraps Moonshine's `MicTranscriber` and puts *finished phrases* on a queue.
  It owns no thread — Moonshine already runs the audio callback and a worker — so it is
  just the queue between their callbacks and your loop. Where a phrase ends is decided
  inside the library from the audio; there is no silence timeout here to tune.
- **`Responder`** streams one chat completion per phrase over a pooled connection, keeping
  the last few turns. Streamed because a 200-token reply delivered in one lump throws away
  exactly the latency the fast recognition bought.

`poll()` never blocks, which is what lets it sit inside a controller's loop:

```python
listener = Listener().start()
while agent.busy:
    agent.step()
    if said := listener.poll():            # None until a phrase is finished
        agent.set_goal(said, {"steps": 300})
```

`Listener.keyterms([...])` re-biases the decoder mid-run, which is worth doing from whatever
the agent is currently looking at — it is most of the difference between "mine the diamond
ore" and "mine the diamond or". `MINECRAFT_KEYTERMS` is the starting list.
`Listener.mute()` stops feeding audio without tearing the model down, for while the agent
is talking back.

## Running next to ROCKET-2

They do not contend where it would hurt. ROCKET-2 is GPU-only and the recognition is
CPU-only, so nothing is taken from the 8 GB the sim is also rendering into. What they share
is the CPU, and Moonshine's threads spend most of their time outside the GIL.

Two orderings matter, both for the same reason a `JarvisVLAClient` is built before the sim:
call `load()` *before* the Minecraft boot, since the first one downloads weights, and reach
the server once before paying for a reset.

## Where it goes wrong

| symptom | cause |
|---|---|
| `OSError: PortAudio library not found` on import | the C library is missing — `conda install -c conda-forge portaudio` |
| a wall of ALSA `Expression ... failed`, then `falling back to 48000 Hz` | harmless, and hidden by default: PortAudio probing a card that does not offer 16 kHz, after which the library resamples. It is written from C to fd 2, so `_quiet_audio` moves the descriptor around the two calls that open the device. `--audio-log` keeps it |
| **you speak and nothing appears** | almost certainly the wrong input device — PortAudio cannot see PipeWire. `--list-devices` shows which one is live; startup prints which was chosen |
| the first run hangs for ~45s | downloading weights, once — this is failure mode 2, call `load()` early |
| the rollout stutters whenever anyone speaks | the chat model and JarvisVLA are on one vLLM |

That last one is the one to watch. JarvisVLA calls its server **once per environment step**,
serially, and is latency-bound in the way [performance.md](performance.md) documents; a chat
completion is hundreds of tokens of decode against its eight. `VOICE_LLM_URL` is deliberately
a separate variable from `JARVISVLA_URL` — point it at a second server, or accept the stutter.

## Configuration

Everything has an environment variable, read by `VoiceConfig.from_env()`.

| variable | default | |
|---|---|---|
| `VOICE_ARCH` | `SMALL_STREAMING` | see the table above |
| `VOICE_LANGUAGE` | `en` | `es de ja ko vi uk zh ar tl` also ship |
| `VOICE_DEVICE` | PipeWire's source, else probed | index or name substring; `--list-devices` measures them |
| `VOICE_AUDIO_LOG` | `0` | `1` to let PortAudio's probe chatter through |
| `VOICE_UPDATE_INTERVAL` | `0.5` | seconds between streaming updates — not phrase length |
| `VOICE_LLM_URL` | `http://127.0.0.1:8000/v1` | the chat server |
| `VOICE_LLM_MODEL` | read from `/v1/models` | |
| `VOICE_TEMPERATURE` | `0.7` | |
| `VOICE_MAX_TOKENS` | `200` | short, because it is being spoken |
