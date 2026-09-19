# ROCKET-2: the (point, interaction, stop) controller

A text-conditioned policy cannot say *which* object you mean: MineCLIP text embeddings
average 0.19 on the 12-task Minecraft Interaction benchmark and score ~0 on anything
targeted (hunt, interact, place). ROCKET-2 replaces the text channel with a pointed goal — a
mask plus an interaction type — and scores ~0.94 on the same benchmark.

That maps onto exactly three fields, which is the whole API here:

```python
from mcagents.agents.rocket2 import Rocket2Agent

agent = Rocket2Agent(sim)                        # sim: obs_size=(224,224) + PrevActionCallback
result = agent.run(
    point=[420, 190],                            # [x, y] on the current frame -> SAM-2 -> mask
    interaction="Mine",                          # Hunt|Mine|Use|Interact|Craft|Switch|Approach
    stop={"item": "log", "count": 3},            # or 200, {"steps": 200}, {"stat": ...}, or a callable
)
print(result)   # item after 143 steps (11.4s), gained oak_log+3, visibility 0.87
```

Run it:

```bash
bash scripts/rocket2.sh                          # launches its own Minecraft
MC_PORT=9000 bash scripts/rocket2.sh             # attaches to a scripts/mc_server.sh holder
bash scripts/rocket2.sh --plan my_plan.json --no-city   # plain terrain, no city at all
```

The plan is a JSON list of goals; the default lives at the top of
[`mcagents/cli/rocket2.py`](../mcagents/cli/rocket2.py). `"point": "click"` lets you click
the target yourself, which is the fastest way to sanity-check a goal before a pointing model
is wired in.

For a one-off there is no need to write a plan file at all — `--goal` takes the description
directly, and repeats for a sequence:

```bash
bash scripts/rocket2.sh --goal "the low white building across the street"
bash scripts/rocket2.sh --goal "the yellow storefront on the corner" \
                        --interaction Approach --stop 400
```

`--stop` takes a step count or the JSON form (`'{"item": "oak_log", "count": 3}'`), and
defaults to 300 steps. Since a *description* is exactly what OWLv2 is weakest at, `--goal`
selects the VLM targeter on its own unless `--targeter` or `$MCAGENTS_TARGETER` says
otherwise — so it needs the server in [Pointing](#pointing) to be up.

## The three fields

**`point`** — `[x, y]` in the current frame, which is `info['pov']` at `render_size`
(640×360 by default), origin top-left. Pass `normalized=True` for `[0..1]` fractions
instead; that is usually what you want from an LLM or a VLM, which does not have to know the
render resolution. SAM-2 (`facebook/sam2.1-hiera-tiny`, via `transformers` — already in this
env, no extra install) turns the point into the mask.

In a plan entry, `point` may also be a *description* — `"tree"`, `"the nearest cow"` — which
`mcagents.perception` resolves against the current frame. See [Pointing](#pointing).

The mask is captured **once per goal** and then frozen. ROCKET-2 tracks the target itself as
the view changes — the "cross-view goal alignment" in its title, and why it is 3–6× faster
than ROCKET-1, which needed a fresh segmentation every 3–10 steps. So the pointing model
runs once per goal (~60–200 steps), not once per frame: a cloud VLM call is affordable there.

**`interaction`** — one of `Hunt / Mine / Use / Interact / Craft / Switch / Approach`, or
`None`. These map to the embedding ids ROCKET-2 was trained with, copied verbatim from its
own demo. Two oddities are upstream's, not bugs to fix: `Use` and `Interact` share id 3, and
id 1 is unused.

**`stop`** — when the goal ends. Shared with JarvisVLA, so a planner that can drive one
controller can drive the other:

| form | meaning |
|---|---|
| `200` or `{"steps": 200}` | a step budget |
| `{"item": "log", "count": 3}` | 3 more of that item in the inventory since the goal started |
| `{"item": "log", "count": 3, "mode": "total"}` | 3 in the inventory in absolute terms |
| `{"stat": "mine_block", "match": "log", "count": 3}` | 3 more on Minecraft's own stat counters (`mine_block`, `kill_entity`, `craft_item`, `pickup`, `use_item`, `damage_dealt`) |
| `{"arrive": {"width": 0.6}}` | the locked target's box is ≥ 60% of the frame wide (`height` works too; `true` = the default width, a bare number = a width). Combines with any row above as "whichever first" |
| `lambda agent: ...` | anything else; gets the agent, returns a bool |

**Approach arrives by default.** An Approach whose `stop` says nothing about arriving —
`null`, a step count, or a dict with no `arrive` — gets `"arrive": {"width": 0.6}` added
(`ROCKET2_ARRIVE_WIDTH`), so it ends when it gets there rather than when its budget runs out.
`"arrive": false` turns that off. A target that already fills the width ends the goal as
`arrived` before any action is taken.

**Approach locks its target.** ROCKET-2 steers by one goal image, and near the target that
image stops resembling the view — which is when it would drift onto a lookalike down the
street. So Approach goals (and any goal with `arrive`) carry an `InstanceLock`
([mcagents/perception/tracking.py](../mcagents/perception/tracking.py)): the instance under
the point, carried by the camera's rotation between updates and re-segmented with SAM-2
every `ROCKET2_TRACK_EVERY` (3) steps, accepted only on mask overlap, plausible size and
similar colour. The description is never re-resolved. If the lock goes unconfirmed for
`ROCKET2_LOCK_GRACE` (20) steps the goal ends as `lost` instead of following the policy.
Mine and Hunt do not lock — their target is supposed to disappear. Tracking costs ~13 ms a
step on average (SAM-2 in bf16, ~35 ms a call); `MCAGENTS_FOV` must match the game's FOV
(default 70) for the camera compensation to be right.

Item names are matched loosely (`mcagents.minecraft.inventory.count_item`): exact first,
then substring, with namespaces stripped — so `"log"` covers `oak_log`/`birch_log`/… and a
planner does not have to know how 1.16.5 spells things. Every goal is also bounded by
`max_steps` (default 600, `ROCKET2_MAX_STEPS`) so a stop condition that never fires cannot
hang the run — which also caps a larger `{"steps": N}`, and says so when a goal starts.

**What a step is.** One `sim.step()`: one game tick (50 ms of game time — the sim does not
advance between steps) and one policy action. In wall time, measured: ~60 ms for Mine/Hunt
(26 ms sim, ~30 ms ROCKET-2 at cfg 0, preview) and ~73 ms for a locked Approach, i.e. 600
steps is 36–45 s. Walking covers ~0.2 blocks a step.

## What comes back

`run()` returns a `Rocket2Result`: `reason` (`item` / `stat` / `steps` / `predicate` /
`arrived` / `max_steps` / `terminated` / `lost` / `cancelled` / `replaced`), `steps`,
`seconds`, `gained` (inventory delta), `visibility`, and `.success` (True only for the first
five). `agent.status()["box"]` is the target's box as frame fractions.

`agent.status()` is the JSON-able snapshot to feed back to the planner each tick. The field
worth building on is **`visibility`** — ROCKET-2's own estimate that its target is still in
view. It is the cheapest "this goal has gone off the rails, re-point me" signal available,
and it costs nothing extra: the model already predicts it. `predicted_point` is where the
model currently thinks the target is, in frame pixels.

For a planner that wants to interleave its own work, step it by hand instead of `run()`:

```python
agent.set_goal(point=[420, 190], interaction="Mine", stop={"item": "log", "count": 3})
while agent.busy:
    agent.step()
    if agent.visibility < 0.2:      # lost it -- ask for a fresh point
        break
```

## Taking goals from a person, live

`--listen` swaps the plan for a queue. Instead of walking a fixed list and exiting, the run
waits for goals to arrive from the voice client and runs each one as it comes:

```bash
MC_PORT=9000 bash scripts/rocket2.sh --listen          # wait for goals
MC_PORT=9000 bash scripts/rocket2.sh --goal "tree" --listen   # ...after this one
```

The goals arrive as plan entries in `/tmp/mcagents-goals` — one file each, oldest first,
put there by `mcagents.local_client` as the model sends them down its second tunnel. See
[docs/local-client.md](local-client.md) for the whole loop and
[mcagents/goals.py](../mcagents/goals.py) for the queue.

**The spool is read before every step, not only between goals.** What arrives decides what
happens to the goal in progress:

| arrives mid-goal | effect | status |
|---|---|---|
| `{"cancel": true}`, or `{"interaction": "None"}` with no [x, y] point | ends now, no-op steps until the next goal; never reaches the targeter | `ended` / `cancelled` |
| `keep_memory: true`, same interaction (and same `id`, if both have one) | merged: lock, policy memory and step count kept; its `stop` replaces the old one, still counted from the goal's start; its point is compared with the lock, never used to retarget | `updated` / `keep_memory` |
| `{"turn": deg}`, `0 < \|deg\| <= 180` | replaces the running goal, which ends first; then turns in place and stands still (below) | `ended` / `replaced`, then `ended` / `turned` |
| a turn out of range, or carrying goal keys | refused; the running goal carries on | `rejected` / `invalid` |
| anything else | replaces the running goal, which ends first | `ended` / `replaced` |

A `keep_memory: true` resend landing within `--followup-window` (5 s) of that goal ending is
refused (`rejected` / `stale_followup`) rather than restarting what was just stopped.
Several goals sent in one message are a plan: they run in order and do not preempt each
other. Every goal's `started`, `ended` and `rejected` go to `/tmp/mcagents-status`
(`MCAGENTS_STATUS_DIR`, `--status-dir`) for the client to send upstream.

**A turn goes around the policy.** "Turn right", "look behind you": there is no target to
point at, so `{"turn": deg}` is not a goal at all. `GoalLoop.turn` clears the policy memory
and calls `Agent.turn_step(yaw)` -- a no-op action with only `camera: [0, yaw]` set -- in equal
pieces of at most `--turn-step` degrees (default 15, `MCAGENTS_TURN_STEP`; 90 is 6 steps, 180
is 12). Positive yaw turns right. MineStudio's env format turns exactly the degrees it is
given, measured 1:1 from 1 to 180 in a single step with no clamp, and pitch and position do
not change; the pieces exist so the published frames show a turn rather than a cut. A turn is
not interrupted -- anything arriving meanwhile is read the moment it ends -- and the goal it
preempted is not resumed. It also works as a `--plan` entry.

Two things it does that a plan loop does not have to:

- **It keeps stepping between goals.** A sim with nothing to do is normally just closed, but
  a run that waits has to keep the world moving: a frozen sim stops publishing frames within
  seconds, and whatever is choosing the next goal is then planning against a picture of a
  world that has stopped. `Agent.idle_step()` is that no-op step. It is paced to
  `--idle-hz` (default 5, `MCAGENTS_IDLE_HZ`, 0 for uncapped): flat out, a wait renders and
  previews at full load for as long as the process lives, which heats a laptop to a crash.
- **It does not record unless asked.** MineStudio's `RecordCallback` holds every frame in RAM
  (~0.7 MB each) and writes the mp4 only on exit. A plan ends; a listener does not, and at
  20 steps a second fills 30 GB in about half an hour. `--record` turns it back on for a
  short session.
- **It survives a goal it cannot run.** A description nothing in frame matches, a targeter
  that errors (a VLM that is down used to kill the process), an unknown interaction, a
  `stop` outside the shared vocabulary — each is skipped with a reason, sent upstream as
  `rejected`, and the next goal is taken. A run driven by a person is not allowed to end because the person
  asked for something odd.

Goals queued before the process started are discarded on the way in: they were said to a
different session, about a world that no longer exists.

## Pointing

Something has to produce `point`. `mcagents.perception` has three answers behind one
`locate(frame, text)` contract:

| backend | what it is | when |
|---|---|---|
| `OwlV2Targeter` (default) | OWLv2 locally, ~600 MB, ~100 ms | bare nouns, no server needed |
| `VLMTargeter` | Qwen2.5-VL on the GPU box over HTTP | phrases ("the nearest tree"); it has actually seen Minecraft |
| `mcagents.gui.pick_point` | a human click | debugging a goal |

Pick with `--targeter vlm` or `MCAGENTS_TARGETER=vlm`. Check a query on a still frame before
trusting it in a run:

```bash
python -m mcagents.cli.locate frame.png "tree"      # writes frame.boxes.png
```

Minecraft is off OWLv2's training distribution, so scores run low — `MCAGENTS_OWL_THRESHOLD`
(default 0.1) is worth tuning against your own frames. Measured on the city spawn frame:
`"street lamp"` 0.19, `"building"` 0.13, `"road"` 0.12, and `"tree"`, `"wall"` and `"door"`
no match at all.

That last one is a trap worth knowing about. [`city.py`](../mcagents/minecraft/city.py) puts
the park *beside* spawn so there would be an oak to point at, but nothing pins which way the
player faces, and in the baked world the first frame is a brick tower across the plaza with
no tree anywhere in it. A goal whose target is not in the first frame is skipped, not fatal —
the run says so and moves to the next goal.

## Watching a run

Two ways, both on by default, matching [JarvisVLA](jarvisvla.md):

| | what it shows | turn it off |
|---|---|---|
| live window | the goal mask, the point you gave, ROCKET-2's own crosshair for where the target is *now*, and the visibility bar | `--no-preview`, `ROCKET2_PREVIEW=0` |
| `logs/rocket2/episode_*.mp4` | the raw game view, no overlay | `--no-record` (already off with `--listen`; `--record` to force) |

The window is drawn after every step, so nothing appears until the first one — the boot, the
world load, the city and the targeter pass all happen before it. It also drops itself when
there is no display to open it on (see *Windows: import order matters* below), which is what
makes the mp4 worth having: an ssh run leaves a file to watch afterwards either way.

Neither is the playable game window. That is `PlayCallback`, and only
[`scripts/play.sh`](../scripts/play.sh) uses it — an agent run drives the sim itself, so
there is nothing to hand a human viewer.

## Measured on this box (RTX 5060 laptop, 8GB)

| | |
|---|---|
| ROCKET-2 forward, `cfg_coef=0` (default) | **33 FPS**, 950 MB VRAM |
| ROCKET-2 forward, `cfg_coef=1.0` | 16 FPS — two forward passes per step |
| SAM-2 tiny | ~455 MB VRAM, once per goal |
| End-to-end with a live Minecraft | ~11 env-steps/s |

The end-to-end number is bounded by MineRL's own step (~50 ms), not by the policy.
`cfg_coef` is the knob worth sweeping per task: it buys goal adherence and costs a forward
pass. ROCKET-2's own gradio demo defaults to 1.0, but it is not driving a live game loop; 0
is the default here for that reason. Set it with `--cfg 1.0` or `ROCKET2_CFG=1.0`.

## Where the code comes from

The ROCKET-2 policy is **not** in the `minestudio` pip package.
[`scripts/setup/vendor_rocket2.py`](../scripts/setup/vendor_rocket2.py) downloads the two
files needed (`model.py`, `cfg_wrapper.py`) from the CraftJarvis/ROCKET-2 repo into
`mcagents/vendor/rocket2/` and applies one patch: timm ≥ 1.0.13 rejects the bare
`timm/vit_base_patch16_224.dino` backbone names hardcoded upstream and wants an `hf-hub:`
prefix. Don't hand-edit the vendored files — re-run the script instead. Checkpoint:
`phython96/ROCKET-2-1x-22w` (187M params, 750 MB), pulled from the Hub on first use;
`ROCKET2_CKPT` overrides it.

## What this does not cover

ROCKET-2 is an *interaction* policy: it needs a target that is visible right now. Two gaps
to plan around:

- **No visible target** — "go explore", "find water", "wander until you see a village". A
  pointed controller has nothing to hold on to; drive exploration with scripted movement or
  hand it to JarvisVLA, whose goal channel is a sentence.
- **Crafting menus** — `Craft` as an interaction type gets the agent to a crafting table; it
  does not drive multi-step GUI recipes. That is exactly what
  [JarvisVLA](jarvisvla.md) was trained on.

## Windows: import order matters

PyAV, which minestudio imports, bundles its own `libxcb` under `av.libs/`. With that loaded
next to the one Qt uses, OpenCV's **first** `cv2.imshow()` never returns — it blocks forever
building the window, with no error and nothing on screen. Later calls are fine once one has
succeeded, so [`mcagents/gui.py`](../mcagents/gui.py) opens and closes a throwaway window at
import time and everything after that works.

The one rule that follows: **import `mcagents.gui` (or `mcagents.agents.rocket2`, which does
it for you) before minestudio.** Get it backwards and nothing hangs — the module says so on
stderr, sets `GUI_READY` False, drops the preview, and `pick_point()` writes the frame to a
temp file and asks for `x y` on the terminal instead. That fallback is also what makes a
click-plan usable over ssh, where there is no display at all.
