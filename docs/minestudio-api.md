# MineStudio environment API — reference for agent coding

Verified against the installed `minestudio` 1.1.6 source
(`~/miniforge3/envs/minestudio/lib/python3.10/site-packages/minestudio/`), not from memory —
file:line references included so you can double-check anything here as the package evolves.
For env setup / GPU rendering / how to launch things, see [setup.md](setup.md), not this file.

## 1. MinecraftSim basics

```python
from minestudio.simulator import MinecraftSim

sim = MinecraftSim(
    action_type="env",        # "env" (flat human-readable actions) or "agent" (VPT-style, DEFAULT)
    obs_size=(224, 224),
    render_size=(640, 360),   # actual Minecraft render resolution
    callbacks=[...],
)
obs, info = sim.reset()
obs, reward, terminated, truncated, info = sim.step(action)
sim.close()
```

**`action_type` default is `"agent"`, not `"env"`** (`entry.py:105`) — easy to miss. Pick based on
what's stepping the env:
- `"env"` — flat dict of individually-meaningful keys (forward/attack/camera/...). What you want
  for hand-written policies, scripted behavior, or manually converting an LLM's plan into
  keypresses.
- `"agent"` — the 2-key VPT-native discretized format (`buttons`, `camera` indices). What a
  VPT-style policy's `get_action()` outputs natively, and what JarvisVLA's action tokens
  decode to.

You can mix them: run the sim as `"env"` and convert a policy's `"agent"`-format output
yourself (see §3) — this is what `PlayCallback` does, and probably what you want for an
LLM-driven agent loop where you also need readable actions for logging/debugging.

## 2. Action space

For `action_type="env"`, `sim.action_space` (`entry.py:285-317`) is a `gymnasium.spaces.Dict`
with exactly these 20 keys:

```
attack, back, forward, jump, left, right, sneak, sprint, use,
hotbar.1 .. hotbar.9, inventory,
camera   # Box(low=-180, high=180, shape=(2,), dtype=float32) — [pitch_delta, yaw_delta]
```
All the non-camera keys are `Discrete(2)` (0/1). `sim.action_space.sample()` gives you a
correctly-shaped random action dict — useful as a template for building your own.

**Gotcha:** `sim.action_space` is *not* the same as the raw underlying MineRL env's action
space. The raw env (`sim.env`) actually accepts 24 keys — the 20 above plus `chat`, `mobs`,
`pickItem`, `voxels` — which some callbacks (`CommandsCallback`, `SummonMobsCallback`,
`VoxelsCallback`) write into directly. `sim.noop_action()` for `action_type="env"`
(`entry.py:257-258`) returns the **raw 24-key noop**, not a noop matching the declared 20-key
`action_space`. In practice: start from `sim.noop_action()` and only set the keys you care
about, rather than hand-building a 20-key dict from scratch.

## 3. Converting between action formats

If you run `action_type="env"` but load a VPT-style policy (ROCKET-2, VPT itself),
its `get_action()` output is in the "agent" format and needs converting:

```python
agent_action, memory = agent.get_action(sim.obs, memory, input_shape="*")
env_action = sim.agent_action_to_env_action(agent_action)   # entry.py:147-180
obs, reward, terminated, truncated, info = sim.step(env_action)
```

(`sim.env_action_to_agent_action` does the reverse, if you ever need it.)

If instead you create `MinecraftSim(action_type="agent")` (the default), you can feed the
policy's output straight to `sim.step()` with no conversion — that's what MineStudio's own
batch-inference path (`inference/generator/mine_generator.py`) does:

```python
obs, info = sim.reset()
memory = None
for _ in range(num_steps):
    action, memory = agent.get_action(obs, memory, input_shape="*")
    obs, reward, terminated, truncated, info = sim.step(action)
```

## 4. Observations

`obs` returned by `reset()`/`step()` has **exactly one key**: `obs["image"]` —
shape `(*obs_size, 3)`, `uint8`, resized from the raw render via `cv2.resize(..., INTER_LINEAR)`
(`entry.py:239`).

**Axis-order gotcha:** `obs_size` is documented/used as `(height, width)`, but it's passed
straight through as `cv2.resize(..., dsize=self.obs_size)`, and OpenCV's `dsize` is
`(width, height)`. Invisible for the default square `(224, 224)`; if you ever pass a
non-square `obs_size`, the dimensions land swapped relative to what the name suggests.

The raw, full-resolution frame (shape driven by `render_size`, e.g. `(360, 640, 3)`) is
**not** in `obs` — it's only in `info["pov"]` (merged in by `_wrap_obs_info`). `info` also
carries everything else MineRL reports: `inventory`, `equipped_items`, `location_stats`,
`voxels`, `mobs`, `health`, `food_level`, `player_pos`, `is_gui_open`, etc. — and it
accumulates: `MinecraftSim` merges each new `info` into a persistent `self.info` rather than
replacing it wholesale, so a key a callback injected earlier stays visible in `info` on
later steps unless overwritten.

## 5. Callbacks — how the sim is actually customized

Callbacks are the extension mechanism for everything: rewards, task setup, recording,
input injection, chat/observation augmentation. `MinecraftCallback` base hooks
(`simulator/callbacks/callback.py`), all pass-through by default — override only what you need:

```python
class MyCallback(MinecraftCallback):
    def before_reset(self, sim, reset_flag: bool) -> bool: return reset_flag
    def after_reset(self, sim, obs, info): return obs, info
    def before_step(self, sim, action): return action
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        return obs, reward, terminated, truncated, info
    def before_close(self, sim): ...
    def after_close(self, sim): ...
```
Call order: `reset()` → `before_reset` → env reset → `after_reset`.
`step()` → `before_step` → env step → `after_step`. Pass instances via
`MinecraftSim(callbacks=[MyCallback(), ...])`.

**This is how a goal channel reaches a policy** — not by changing `step()`'s signature, but
by having a callback stuff it into `obs` each step (`PrevActionCallback` is exactly this).
Since callbacks can add arbitrary keys to `obs`/`info`, it is also how you would expose your
own agent's internal state for logging without touching `MinecraftSim` itself.

Built-in callbacks worth knowing about (`simulator/callbacks/`):

| Callback | Purpose |
|---|---|
| `SpeedTestCallback(interval)` | Prints steps/sec periodically — what `mcagents.cli.bench` measures by hand. |
| `RecordCallback(record_path, ...)` | Records episode video/actions/infos to disk. |
| `RewardsCallback(reward_cfg)` | Computes reward from in-game events. |
| `TaskCallback(task_cfg)` | Assigns/manages a task. |
| `CommandsCallback(commands)` | Runs raw Minecraft commands at lifecycle events. |
| `InitInventoryCallback(...)` | Sets starting inventory. |
| `FastResetCallback(...)` | Faster reset (biome/teleport/time/weather) vs. the full hard reset. |
| `JudgeResetCallback(time_limit)` | Auto-resets on timeout/termination. |
| `PrevActionCallback()` | Adds the previous action into the observation. |
| `PlayCallback(agent_generator=None)` | Human/agent interactive GUI play — what `play_minecraft.py` uses. |

## 6. Loading a policy

Every policy in MineStudio's gallery is a `MinePolicy` and loads the same way — the standard
`huggingface_hub.PyTorchModelHubMixin` method, repo id or local path:

```python
from minestudio.models import VPTPolicy

policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_for_shoot_animals_2x").to("cuda")
policy.eval()
```

`get_action` signature (`models/base_policy.py:120-127`, inherited unmodified):
```python
action, state_out = policy.get_action(
    input,                     # dict — "image", plus whatever goal channel the policy takes
    state_in,                  # recurrent memory; None on first call, then feed back state_out
    deterministic=False,
    input_shape="*",           # "*" = single unbatched instance (auto-wrapped to (1,1,...))
)
```
`input_shape="*"` is what you want for a live step-by-step loop; the default `"BT*"` expects
you to supply batch+time dims yourself (for offline batch inference).

The goal channel is the part that differs per policy, and it usually arrives through the
*observation* rather than through `get_action` — ROCKET-2 reads `obs["cross_view"]` (a frame,
a mask and an interaction id) and `obs["env_prev_action"]`, which is why its sim needs a
`PrevActionCallback`. See `mcagents/agents/rocket2.py` for that wiring, and
`mcagents/vendor/rocket2/model.py` for the policy itself; ROCKET-2 is not in the pip package.

## 7. Practical notes for an LLM-driven agent loop

- **Where the "LLM" plugs in**: at the goal channel, not in the step loop. Both controllers
  here take a whole goal (a point plus an interaction type, or a sentence) and run it to a
  stop condition over tens or hundreds of steps, so the planner is called once per goal, not
  once per frame. `mcagents/agents/base.py` is that interface.
- Real-time pacing: `PlayCallback.after_step` throttles to `GUIConstants.MINERL_FRAME_TIME`
  (`simulator/utils/constants.py`) for human play; for an autonomous agent loop you likely
  don't want that sleep — step as fast as the sim/policy allow (bounded by the ~30-38
  steps/sec measured with GPU rendering in [setup.md](setup.md) §6).
- Reset is expensive (real JVM/world round-trip, see the "why does reset take so long"
  discussion — not reproduced here, ask if you need it again) — design the agent loop to
  avoid unnecessary resets, e.g. one long episode with periodic `JudgeResetCallback`-style
  soft resets rather than resetting per LLM call.
