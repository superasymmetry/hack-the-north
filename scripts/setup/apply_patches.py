"""
Applies every source patch this project needs on top of a fresh `pip install minestudio`.

Why this exists: minestudio 1.1.6 ships with a handful of real bugs (not our code, not our
config) that only surface once you actually try to use GPU rendering and the interactive
play GUI. Each one is explained in detail in docs/setup.md. This script makes those fixes
reproducible instead of living only as prose -- run it once after any fresh
`pip install minestudio` (new machine, new conda env, SLURM node, minestudio upgrade) and it
re-applies everything. Safe to run multiple times: each patch checks whether it is already
applied and skips itself.

Usage:
    conda activate ./.conda-env
    python scripts/setup/apply_patches.py
"""
import sys
from pathlib import Path

import minestudio

MINESTUDIO_DIR = Path(minestudio.__file__).parent

PATCHES = [
    {
        "name": "gpu_utils.py: missing `cuda` import silently forces CPU rendering",
        "file": MINESTUDIO_DIR / "simulator/minerl/env/gpu_utils.py",
        "old": "    try:\n        call_and_check_error(cuda.cuInit)(0)",
        "new": "    try:\n        from cuda import cuda\n        call_and_check_error(cuda.cuInit)(0)",
    },
    {
        "name": "constants.py: pyglet.canvas was renamed to pyglet.display in pyglet 2.1",
        "file": MINESTUDIO_DIR / "simulator/utils/constants.py",
        "old": "        screen = pyglet.canvas.get_display().get_default_screen()",
        "new": "        screen = pyglet.display.get_display().get_default_screen()",
    },
    {
        "name": "gui.py: TextLayout.update(x=,y=) doesn't exist in any published pyglet",
        "file": MINESTUDIO_DIR / "simulator/utils/gui.py",
        "old": "        layout.update(x=self.window.width//2, y=self.window.height//2)",
        "new": "        layout.x = self.window.width//2\n        layout.y = self.window.height//2",
    },
    {
        "name": "gui.py: force a legacy/compatibility GL context (imgui needs GL_ALPHA + double buffering)",
        "file": MINESTUDIO_DIR / "simulator/utils/gui.py",
        "old": (
            "        if self.show_info:\n"
            "            self.window = self.pyglet.window.Window(\n"
            "                width = self.constants.WINDOW_WIDTH,\n"
            "                height = self.constants.INFO_HEIGHT + self.constants.FRAME_HEIGHT,\n"
            "                vsync=False,\n"
            "                resizable=False\n"
            "            )\n"
            "        else:\n"
            "            self.window = self.pyglet.window.Window(\n"
            "                width = self.constants.WINDOW_WIDTH,\n"
            "                height = self.constants.FRAME_HEIGHT,\n"
            "                vsync=False,\n"
        ),
        "new": (
            "        # imgui's pyglet integration uses legacy fixed-pipeline GL calls (e.g. GL_ALPHA\n"
            "        # textures) that only exist in a compatibility profile. Modern GL drivers default\n"
            "        # to a core profile, so request an old (pre-3.2, always-compatibility) version\n"
            "        # explicitly to guarantee those calls work.\n"
            "        legacy_gl_config = self.pyglet.gl.Config(major_version=2, minor_version=1, double_buffer=True)\n"
            "        if self.show_info:\n"
            "            self.window = self.pyglet.window.Window(\n"
            "                width = self.constants.WINDOW_WIDTH,\n"
            "                height = self.constants.INFO_HEIGHT + self.constants.FRAME_HEIGHT,\n"
            "                vsync=False,\n"
            "                resizable=False,\n"
            "                config=legacy_gl_config,\n"
            "            )\n"
            "        else:\n"
            "            self.window = self.pyglet.window.Window(\n"
            "                width = self.constants.WINDOW_WIDTH,\n"
            "                height = self.constants.FRAME_HEIGHT,\n"
            "                vsync=False,\n"
            "                config=legacy_gl_config,\n"
        ),
    },
    {
        "name": "play.py: command-mode (Esc) busy-loop flips the window with nothing redrawn -> flashing",
        "file": MINESTUDIO_DIR / "simulator/callbacks/play.py",
        "old": (
            "        if 'ESCAPE' in released_keys:\n"
            "            time_count = 0 # Renamed variable to avoid conflict with time module\n"
            "            while True:\n"
            "                self.gui.window.dispatch_events()\n"
            "                self.gui.window.switch_to()\n"
            "                self.gui.window.flip()\n"
            "                current_released_keys = self.gui._capture_all_keys() # Use a different variable name\n"
            "                time_count += 1\n"
            "                if len(current_released_keys) > 0:\n"
            "                    released_keys = current_released_keys # Update the original set if needed\n"
            "                    break\n"
        ),
        "new": (
            "        if 'ESCAPE' in released_keys:\n"
            "            while True:\n"
            "                self.gui.window.dispatch_events()\n"
            "                current_released_keys = self.gui._capture_all_keys() # Use a different variable name\n"
            "                if len(current_released_keys) > 0:\n"
            "                    released_keys = current_released_keys # Update the original set if needed\n"
            "                    break\n"
            "                time.sleep(0.01)\n"
        ),
    },
    {
        # Not a bug -- a constant that has to be configurable for gaze selection to work.
        # The window is 640*SCALE wide, and SCALE is hardcoded to 2, i.e. a 1280x810 view of
        # the game. Gaze accuracy is an *angle*, so how many frame pixels it costs depends
        # entirely on how large the view is on screen: ~104 screen px of error is 52 frame px
        # at SCALE=2 and 26 at SCALE=4. See docs/gaze.md.
        "name": "constants.py: GUI window scale hardcoded to 2 (gaze needs a bigger view)",
        "file": MINESTUDIO_DIR / "simulator/utils/constants.py",
        "old": "        self.SCALE = 2\n",
        "new": '        self.SCALE = int(__import__("os").environ.get("MCAGENTS_GUI_SCALE", 2))\n',
    },
]


def main():
    exit_code = 0
    for patch in PATCHES:
        path: Path = patch["file"]
        name = patch["name"]
        if not path.exists():
            print(f"[SKIP] {name}\n       file not found: {path}")
            exit_code = 1
            continue

        text = path.read_text()
        if patch["new"] in text:
            print(f"[OK]   {name}\n       already applied")
            continue
        if patch["old"] not in text:
            print(f"[WARN] {name}\n       neither old nor new text found in {path}")
            print(f"       minestudio version may have changed this file — check manually")
            exit_code = 1
            continue

        path.write_text(text.replace(patch["old"], patch["new"], 1))
        print(f"[FIX]  {name}\n       patched {path}")

    if exit_code == 0:
        print("\nAll patches applied (or already were).")
    else:
        print("\nSome patches need manual attention — see WARN/SKIP lines above.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
