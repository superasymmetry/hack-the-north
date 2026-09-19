# Dev environment setup — MineStudio on this machine

Machine: Ubuntu 24.04 (Noble), NVIDIA RTX 5060 Laptop GPU (Blackwell, sm_120, 8GB VRAM),
hybrid Intel/NVIDIA (PRIME) laptop. X11 session (gdm3/Xorg on `:1`).

**The conda env lives inside this repo, at `./.conda-env`** (a conda *prefix* env, not a
named one) — not in the default `~/miniforge3/envs/`. Activate it with the full path:
```bash
conda activate /home/stringbot/Documents/Github/htn/.conda-env
```
(History: it was first created as a named env `minestudio` in the default central location,
then migrated project-local via `conda create --clone minestudio -p ./.conda-env` once we
realized we wanted the whole project self-contained. The commands below describe the
*original* central-env creation for narrative accuracy, but if you're setting this up fresh
on a new machine, just create it project-local directly — see the box below — there's no
reason to do the central-then-clone dance again.)

**Reproducing on a new machine / SLURM node, from scratch:**
```bash
# 1. conda (see §1 if you don't have it)
conda create -p /path/to/this/repo/.conda-env -c conda-forge python=3.10 openjdk=8 -y
conda activate /path/to/this/repo/.conda-env
pip install minestudio

# 2. apply this project's fixes on top (pip version pins + source patches — see §5/§8
#    for the *why*, this just mechanically applies them; idempotent, safe to re-run)
bash scripts/setup/install.sh

# 3. system rendering deps — see §3/§4, these are apt/system-level, not conda-scoped

# (step 2 also vendors ROCKET-2's policy, which does not ship with minestudio — see
#  rocket2.md. The ROCKET-2 / SAM-2 / OWLv2 checkpoints download on first run.)
```

## 1. Conda

No conda was present on the machine. Installed Miniforge (conda-forge default channel):

```bash
curl -fsSL -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

## 2. Env + JDK 8 + MineStudio

Originally created centrally (see note above for why this later moved project-local):
```bash
conda create -n minestudio -c conda-forge python=3.10 openjdk=8 -y
conda activate minestudio
pip install minestudio
```

- JDK: OpenJDK 8.0.472 (Zulu build), installed via conda-forge, scoped entirely to the env
  (originally `~/miniforge3/envs/minestudio/bin/java`, now
  `./.conda-env/bin/java` inside this repo). System `/usr/bin/java` untouched either way.
- `minestudio` version installed: **1.1.6**
- Pulled in **torch 2.8.0+cu128** — confirmed working with the RTX 5060's Blackwell
  (sm_120) compute capability; no compatibility issue despite it being a very new GPU arch.

## 3. System rendering deps (apt)

The MineStudio docs list `libegl1-mesa` — **that package no longer exists on Ubuntu 24.04**
(renamed upstream). Use `libegl-mesa0` instead:

```bash
sudo apt update
sudo apt install -y xvfb mesa-utils libegl-mesa0 libgl1-mesa-dev libglu1-mesa-dev xauth xterm
```

## 4. VirtualGL 3.1

```bash
curl -fsSL -o virtualgl_3.1_amd64.deb \
  "https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download"

# The .deb still hard-depends on the old `libegl1-mesa` name (see step 3) — force past
# that single stale dependency check (libegl-mesa0, already installed, is the same lib
# under its current name):
sudo dpkg -i --ignore-depends=libegl1-mesa virtualgl_3.1_amd64.deb
# Do NOT run `apt -f install` afterward — apt doesn't know about the --ignore-depends
# override and will see the "unsatisfied" dependency and remove the package again.
```

### Deliberately skipped: `vglserver_config`

The official docs' default flow is `sudo service gdm stop` -> `vglserver_config` ->
`sudo service gdm start`. That's a **multi-user security lockdown** step (restricts the X
server to a `vglusers` group). Skipped it here because:

- This is a single-user dev box — the restriction step is not needed for the current user
  to access the display.
- Stopping gdm would have killed the live desktop session (and the terminal/IDE running
  in it).

Confirmed the current user already has valid X11 auth (`xauth list`) for `:1` without any
of that.

### Rendering backend: VirtualGL EGL, not GLX-to-a-second-X-server

Tried and rejected: pointing VirtualGL's "3D X server" at the live desktop's `:1` via
NVIDIA PRIME render offload (`__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`)
— this worked, but ties the setup to having a live desktop session, which won't exist on
headless SLURM nodes.

**What's actually used** (and what MineStudio's own `launchClient.sh` defaults to,
independently — see `minestudio/simulator/minerl/env/launchClient.sh`):

```bash
export PATH="${PATH}:/opt/VirtualGL/bin"
Xvfb :99 -screen 0 1280x1024x24 &      # headless 2D display for the app/window
export DISPLAY=:99
# GL rendering itself goes straight to the GPU via VirtualGL's EGL backend
# (talks to /dev/dri directly, no second X server needed):
#   vglrun -d egl <app>
```

Verified with `vglrun -d egl glxinfo` / `glxspheres64`:

```
direct rendering: Yes
OpenGL vendor string: NVIDIA Corporation
OpenGL renderer string: NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2
```

No `llvmpipe`/`softpipe` anywhere — confirmed hardware rendering, not the Xvfb/CPU fallback.

### To run the simulator with GPU rendering

```bash
export DISPLAY=:99   # headless Xvfb must be running (see above)
MINESTUDIO_GPU_RENDER=1 python -m minestudio.simulator.entry
```

`MINESTUDIO_GPU_RENDER=1` makes MineStudio's internal launcher pick `vglrun -d egl`
(or a specific `/dev/dri/by-path/...` node resolved from `CUDA_VISIBLE_DEVICES`, see
`minestudio/simulator/minerl/env/gpu_utils.py`) instead of the CPU/`xvfb-run` path.

### For SLURM reproduction later

No live desktop session will exist on compute nodes. The Xvfb + `vglrun -d egl` pattern
above should work unmodified as long as:
- The node has an NVIDIA GPU with `/dev/dri` render nodes exposed to the job, and
- VirtualGL + Xvfb are installed (or baked into a container/module).

No `vglserver_config` / display-manager step is needed there either — it was never needed
here.

## 5. Simulator gate

```bash
export PATH="${PATH}:/opt/VirtualGL/bin"
export DISPLAY=:99            # headless Xvfb must already be running (see step 4)
export MINESTUDIO_GPU_RENDER=1
python -m minestudio.simulator.entry -y   # -y skips the HF engine-download confirmation prompt
```

First run downloads the compiled simulator engine (`CraftJarvis/SimulatorEngine` on
Hugging Face — **not Minecraft/Mojang**, no account needed) to `/tmp/MineStudio` and
extracts it (a prebuilt jar — no local Gradle build was actually triggered on this run,
the "first launch runs Gradle" case applies if building from source instead of the
pip-distributed prebuilt engine).

The `entry.py` `__main__` block doubles as a smoke test: resets the env, steps 100 times
with a `SpeedTestCallback`, prints FPS.

### Two upstream bugs found and fixed (both in the installed `minestudio` 1.1.6 package)

1. **Missing `psutil` dependency.** `minestudio.simulator.minerl.utils.process_watcher`
   imports `psutil` directly but it isn't declared as a package dependency.
   Fix: `pip install psutil`.

2. **GPU device detection always silently fell back to CPU**, even with
   `MINESTUDIO_GPU_RENDER=1` set and a working GPU. Root cause, in
   `minestudio/simulator/minerl/env/gpu_utils.py`:
   - The `__main__` block calls `cuda.cuInit(0)` but never imports `cuda` in that scope
     (only imported locally inside two *other* functions) → `NameError`, caught by a bare
     `except:`, which prints `"cpu"` unconditionally.
   - Fixed by adding `from cuda import cuda` before the `cuInit` call. That surfaced a
     **second**, more fundamental problem: `pip install minestudio` pulls the latest
     `cuda-python` (13.3.1), which restructured its API — `cuda.cuda`/`cuda.cudart` are
     gone, moved to `cuda.bindings.driver`/`cuda.bindings.runtime`. MineStudio's code is
     written against the old (pre-13.x) layout and the package doesn't pin a compatible
     version.
   - Fix: `pip install "cuda-python==12.9.7"` (last 12.x release; matches this machine's
     driver's CUDA 12.9). This deprecation still emits `FutureWarning`s (harmless) but
     `cuda.cuda`/`cuda.cudart` work.

   Patched file (this file lives inside the conda env's site-packages, so it needs
   reapplying after any fresh `pip install minestudio` on the SLURM cluster too):
   `minestudio/simulator/minerl/env/gpu_utils.py`, in the `if __name__ == "__main__":`
   block:
   ```python
   try:
       from cuda import cuda        # <-- added; was missing
       call_and_check_error(cuda.cuInit)(0)
   except:
       print("cpu")
       exit(0)
   ```

### How to confirm the simulator is actually using the GPU (not silently CPU)

Look for this line in the simulator's stdout:
```
INFO: Starting Minecraft process with device: /dev/dri/card1
```
If it instead says `device: cpu`, GPU rendering silently failed — check the two fixes
above first (this was the actual failure mode hit during setup: FPS ~19-21 on the
CPU-fallback path vs. ~28-32 once genuinely on `/dev/dri/card1`).

## 6. Benchmark

`python -m mcagents.cli.bench`: resets `MinecraftSim(action_type="env")`, steps 500 times
with random actions sampled from `sim.action_space`, times it, prints steps/sec. No tuning
applied. It is the ceiling any controller runs into.

```bash
export PATH="${PATH}:/opt/VirtualGL/bin"
export DISPLAY=:99
export MINESTUDIO_GPU_RENDER=1
python -m mcagents.cli.bench
```

**Result: 500 steps in 13.15s → 38.03 steps/sec.** Confirmed GPU rendering active
(`device: /dev/dri/card1`) during the run, not CPU fallback.

## 7. Running a controller

Both controllers boot the simulator the same way — see
[rocket2.md](rocket2.md) and [jarvisvla.md](jarvisvla.md) for the controllers themselves, and
`scripts/env.sh` for the environment every runner script sources.

```bash
bash scripts/rocket2.sh          # a pointed goal, policy on this GPU
./scripts/jarvisvla.sh           # an English instruction, policy on the GPU box
```

### Two operational gotchas hit on a fresh day (after a machine reboot / `/tmp` cleared)

Neither is a bug, both will recur on SLURM after any node restart — worth automating there
rather than hitting them manually each time:

1. **`/tmp/MineStudio` (the downloaded simulator engine) gets wiped on every reboot.**
   Not the 30d age policy — `/usr/lib/tmpfiles.d/tmp.conf` has `D /tmp`, and type `D`
   means *empty the directory at boot*. `MinecraftSim(...)` then blocks on an interactive
   "download it from huggingface (Y/N)?" prompt; with no stdin attached that is an
   `EOFError`. Two-part fix, both now in place:
   - `scripts/env.sh` exports `MINESTUDIO_DIR="$HOME/.minestudio"` so the 445 MB
     `mcprec-6.13.jar` lives outside `/tmp` and survives reboots (see
     `minestudio/utils/temp.py` — the env var is read there, default is
     `tempfile.gettempdir()/MineStudio`).
   - `mcagents.minecraft.session.Session.open()` calls `check_engine(skip_confirmation=True)`
     (from `minestudio.simulator.entry`) before constructing `MinecraftSim`, so a missing
     engine re-downloads instead of prompting. Equivalent one-off: `python -m
     minestudio.simulator.entry -y` per §5.
   Note the model weights are *not* affected — they are in `~/.cache/huggingface`.

2. **The headless `Xvfb :99`** (started manually, not as a service) does not survive a
   reboot. Without it, the Minecraft client fails fast with a Java-level crash:
   `java.lang.IllegalStateException: Failed to initialize GLFW, errors: GLFW error during
   init` — worth recognizing this specific error as "Xvfb isn't running", not a rendering
   regression. Fix: `Xvfb :99 -screen 0 1280x1024x24 &` before running anything with
   `DISPLAY=:99`. On SLURM this should be a proper per-job/per-node startup step (or a
   systemd service), not a manually-run background command.

## 8. Interactive human play

`python -m mcagents.cli.play` (`bash scripts/play.sh`): MineStudio's built-in `PlayCallback` (human
keyboard/mouse control, no agent). Note: the official tutorial
(`minestudio/tutorials/simulator/test_play.py`) calls `sim.step(sim.action_space.sample())`
in its loop — that actually *bypasses* human control, since `PlayCallback.before_step`
only reads real input when the action passed in is `None` or a string (see
[play.py:130](minestudio/simulator/callbacks/play.py#L130)). Use `sim.step(None)` instead.

```bash
export PATH="${PATH}:/opt/VirtualGL/bin"
export DISPLAY=:1            # the LIVE desktop, not the headless :99 — PlayCallback opens
                              # a real window (pyglet) that a human needs to see and click into
export MINESTUDIO_GPU_RENDER=1
python -m mcagents.cli.play
```

Controls: standard Minecraft WASD/mouse once in human mode (default). Extra bindings:
`C` capture mouse, `Ctrl+C` close window, `Esc` command mode, `L` switch human/agent control.

### Three more upstream bugs found and fixed (installed `minestudio` 1.1.6 package)

MineStudio's debug/play GUI (`minestudio/simulator/utils/gui.py`,
`minestudio/simulator/utils/constants.py`) is built on `imgui` + `pyglet` and was clearly
written against an unpinned, likely pre-release/dev snapshot of `pyglet` — no single
published `pyglet` version satisfies everything it assumes. All three fixes below are in
the conda env's site-packages and need reapplying after any fresh `pip install minestudio`.

1. **`pyrender` hard-pins `pyglet==1.4.0b1`** (a 2018-era beta), which is what
   `pip install minestudio` resolves to by default — but MineStudio's own GUI code needs a
   pyglet 2.x API. Since `pyrender` isn't used by the play path, force pyglet forward:
   ```bash
   pip install "pyglet==2.1.16"
   ```
   (`pip` will complain about the `pyrender` conflict — harmless, `pyrender` isn't touched
   by this script.)

2. **`constants.py:71`** called `pyglet.canvas.get_display()` — the `canvas` module was
   renamed to `display` in pyglet 2.1. Fixed:
   ```python
   screen = pyglet.display.get_display().get_default_screen()   # was pyglet.canvas...
   ```

3. **`gui.py:347`** called `layout.update(x=..., y=...)` on a `pyglet.text.layout.TextLayout`
   — that convenience method doesn't exist in any published pyglet release (2.0.x or
   2.1.x), only in whatever dev snapshot MineStudio was written against. Fixed by setting
   the position attributes directly, which do exist:
   ```python
   layout.x = self.window.width//2   # was layout.update(x=..., y=...)
   layout.y = self.window.height//2
   ```

4. **`gui.py` `create_window()`** — `imgui`'s pyglet integration (`PygletFixedPipelineRenderer`)
   uses legacy fixed-pipeline GL calls (e.g. a `GL_ALPHA` texture format via `glTexImage2D`)
   that only exist under an OpenGL **compatibility** profile. Pyglet's default `Window()`
   config on this machine negotiated a **core** profile, which doesn't have those calls →
   `GLError: invalid value`. Fixed by explicitly requesting a legacy (pre-3.2, always
   full/compatibility) GL context, since core vs. compatibility profiles only exist from
   GL 3.2 onward — requesting an older version sidesteps the split entirely:
   ```python
   legacy_gl_config = self.pyglet.gl.Config(major_version=2, minor_version=1, double_buffer=True)
   self.window = self.pyglet.window.Window(..., config=legacy_gl_config)
   ```
   Note `double_buffer=True` was added explicitly: pyglet's `Config.double_buffer` defaults
   to `None` (unspecified — left to driver negotiation) even when not otherwise touched.
   Forcing the unusual legacy-GL-version request above without pinning this left the visual
   selection ambiguous and got a single-buffered visual on this machine, which caused the
   window to visibly flash every frame (`window.clear()` wiping the buffer actually being
   displayed, before the next frame's draw calls landed). Pinning `double_buffer=True`
   fixed it.

5. **`play.py`'s "command mode" (Esc) key-handling loop flashed continuously** while active.
   `process_keys()`'s `ESCAPE` branch busy-waits for the next key release by calling
   `window.switch_to()` + `window.flip()` on every loop iteration with nothing redrawn in
   between — so it just repeatedly swaps the front/back buffer between two stale, unrelated
   frames, which reads as flashing, for as long as you stay in command mode. It also spun
   the CPU at 100% the whole time. Fixed by dropping the pointless `switch_to()`/`flip()`
   calls (no draw call between them ever accomplished anything) and adding a small sleep:
   ```python
   if 'ESCAPE' in released_keys:
       while True:
           self.gui.window.dispatch_events()
           current_released_keys = self.gui._capture_all_keys()
           if len(current_released_keys) > 0:
               released_keys = current_released_keys
               break
           time.sleep(0.01)
   ```
   File: `minestudio/simulator/callbacks/play.py`.

## 9. Reset latency — why "Resetting environment..." sits there, and the JVM-reuse option

Measured from 25 runs' worth of `logs/mc_*.log`, `sim.reset()` breaks down as:

| Phase | Time |
| --- | --- |
| JVM + Forge/OptiFine boot → `***** Start MalmoEnvServer` | ~11–14 s |
| `Received mission init` → `Starting integrated minecraft server` | ~2–3 s |
| server init → `Preparing start region` | ~6 s |
| `Preparing start region` → `MineRLAgent0 joined the game` | ~12–13 s |
| `num_empty_frames=20` no-op steps | ~0.7 s |

≈ 33–38 s total, all of it under the GUI's "Resetting environment..." message, which
`PlayCallback.before_reset` puts up at the very top of `MinecraftSim.reset()`.

Worth knowing: every one of those runs logged the *same* spawn,
`MineRLAgent0[...] logged in ... at (-3009.5, 71.0, -5572.5)`. `MinecraftSim(seed=0)` is
fixed and `HumanSurvival` asks for `<DefaultWorldGenerator forceReset="true"/>`, so the
last ~19 s regenerates a byte-identical world from scratch on every single run.

### The JVM-reuse option (implemented)

`mcagents/minecraft/instance.py` holds one Minecraft open across runs, removing the
~11–14 s boot:

```bash
bash scripts/mc_server.sh            # terminal 1, leave running; prints the port
MC_PORT=9000 bash scripts/rocket2.sh # terminal 2, as often as you like
```

`EnvConfig` reads `MC_PORT` (`--mc-port` also works); unset, the runner launches Minecraft
in-process exactly as before. How it works:
`MinecraftSim.reset()` → `_setup_instances()` → `InstanceManager.get_instance()` returns
the first *unlocked* entry in `_instance_pool` before it considers launching anything
(`malmo.py:191`), so `instance.attach()` just puts an `existing=True` instance in that
pool first. MineRL never kills an instance it did not launch (`_destruct` is guarded by
`not self.existing`, `malmo.py:652`), so `sim.close()` leaves the holder's Minecraft alone.

Two upstream snags this works around, both worth knowing if you touch this code:

- `InstanceManager.add_existing_instance(port)` — the obvious API — is broken as shipped:
  it calls `MinecraftInstance(port=..., existing=True)` without the required `working_dir`
  positional (`malmo.py:255` vs `:350`) and raises `TypeError`. `attach()` constructs the
  instance directly instead.
- That `working_dir` must be a scratch dir of the *client's*, not the holder's.
  `MinecraftInstance.__init__` registers it with MineStudio's garbage collector, which
  `rmtree`s it once the registering process exits (`database_manager.py:80`) — pointing it
  at the holder's directory would delete the running instance's runtime under it.

### The world-cache option (implemented)

Reusing the JVM leaves the ~19 s of world generation. `mcagents/minecraft/world.py`
removes most of that by loading a world saved once instead of regenerating it:

```bash
bash scripts/build_world.sh   # once — the slow generation, writes worlds/plains.zip
bash scripts/rocket2.sh       # picks it up automatically from then on
```

`EnvConfig` reads `MC_WORLD` (default `worlds/chiangtung.zip`, or `worlds/plains.zip` while that is absent); if the file is missing the runner
says so and generates a world as before. Measured against the same
attached instance: **22.7 s → 9.0 s**, and in the Minecraft log mission-init → `joined the
game` goes 21 s → 6 s.

Mechanism: `HumanSurvival(load_filename=…)` → `LoadWorldAgentStart` → `<LoadWorldFile>` in
the mission XML (`human_survival_specs.py:15`, `handlers/agent/start.py:278`).
`MinecraftSim` swallows the kwarg (`entry.py:127`), but `EnvSpec.reset()` rebuilds the
handler lists at the top of every `env.reset()`, so setting `sim.env.task.load_filename`
on the live task is enough — no patching of the installed minestudio.

Engine side (from `javap` on `mcprec-6.13.jar`, since none of this is documented):
`EnvServer.loadOrCreateWorld` calls `getSaveFile(missionInit)`, and when that is non-null it
runs `ReplaySender.loadWorldFromZip(path)` *instead of* `createNewWorld`. `loadWorldFromZip`
unzips into `$TMPDIR/<random hex>`, takes the world name from `entries[0].split("/")[2]`,
then loads `<extract root>/saves/<name>`. Those two facts together pin the zip layout down
to `./saves/<world>/…` — the leading `.` is what makes index 2 land on the world name.
`WorldCache.pack()` builds exactly that (hand-rolled `ZipInfo`, because
`zipfile.write()` normalizes `./` away).

Three consequences of taking the load branch:

- **The zip is the seed.** `MinecraftSim(seed=…)` and `preferred_spawn_biome` are both read
  inside `createNewWorld`, the branch that is now skipped, so neither does anything while a
  world is loaded. Rebuild the cache to change worlds. The spawn biome is therefore a
  property of the cache (`WorldCache(biome=…)`, default `"plains"`), which `build()` passes
  to the one generating run and which also names the cache file — so changing it changes the path, and a stale world
  from the previous biome cannot be picked up by accident.
- Every load leaves its `$TMPDIR/<hex>` extraction behind — ~4 MB per reset. Harmless here
  (`D /tmp` empties it at boot, §7) but not on a node where `/tmp` persists.
- Player state travels inside the world. `WorldCache.build()` captures right after a
  reset, so the save is pristine; capturing later would bake in whatever the agent had done.

`<FileWorldGenerator>` looks like the more natural fit and MineRL ships a Python handler for
it, but the jar has only the schema class and no implementation — `<LoadWorldFile>` is the
path that actually works.

### Both together

| | reset |
| --- | --- |
| stock | ~33–38 s |
| `MC_PORT` only | ~21–23 s |
| `MC_PORT` + `worlds/plains.zip` | **~9 s** |
