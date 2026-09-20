"""Track the eyes, and publish where they are looking.

    ./scripts/gaze.sh                 # aim the camera, calibrate, then stream
    ./scripts/gaze.sh --check         # ...and print every sample, to see it working
    ./scripts/gaze.sh --no-calibrate  # reuse the last calibration

This runs on its own, in its own virtualenv, and talks to the game only through
[mcagents/gaze.py](../gaze.py) -- one small file on a tmpfs, newest sample wins. That is not
architectural taste, it is a hard requirement: the tracker wants numpy 2, opencv 5 and
mediapipe, and the process running Minecraft has numpy 1.26 and opencv 4.8 and would not
survive any of them. Nothing here imports torch, minestudio or anything else of the game's.

The tracker is [GazeFollower](https://github.com/GanchengZhu/GazeFollower), which reports
about 1.1 cm of error after calibration -- roughly 100 pixels on this laptop's screen. That
sounds fatal for picking out a Minecraft block and is not, for one reason: error is an
*angle*, so what it costs in frame pixels depends on how large the game is drawn. At
MCAGENTS_GUI_SCALE=4 the view is 2560 px wide for a 640 px frame, and 100 screen pixels
become 25 frame pixels. Then SAM-2 snaps that to an object and the selector holds onto it.
See docs/gaze.md.

Calibration is per person, per seating position, and drifts if you move the laptop. It takes
about half a minute, and `--check` is how you tell whether it is still good.

GazeFollower does not keep it: its `save_model` is a stub, so a restart means calibrating
again, which during an evening of restarting the game is most of an evening of looking at
dots. The model is a ridge regression -- one small matrix -- so this saves and reloads that
matrix itself. Run `--calibrate` when you have moved, and it is thrown away and redone.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from mcagents import gaze

#: Where the fitted calibration is kept between runs. Under the user's cache rather than in
#: the repo: it describes one person in one chair in front of one screen, and belongs to
#: this machine in the way a world seed does not.
DEFAULT_CALIBRATION = Path(
    os.environ.get("MCAGENTS_GAZE_CALIBRATION",
                   Path.home() / ".cache" / "mcagents" / "gaze-calibration.npz"))

#: How old a saved calibration may be before it is worth mentioning. Not enforced -- how
#: much it has drifted depends on whether you moved, which the clock cannot know, and
#: `--check` answers in five seconds.
STALE_CALIBRATION = 12 * 3600

#: The unhelpful half of an ImportError here, replaced with the thing to actually do.
MISSING = """the gaze tracker is not installed in this environment.

    bash scripts/setup/install_gaze.sh

It goes in its own virtualenv (.gaze-env) on purpose -- it needs numpy 2, opencv 5 and
mediapipe, and installing those next to the simulator would break the simulator."""


def build(multiprocessing: bool = False):
    """The tracker, or a usage error explaining how to get one.

    In-process by default, unlike GazeFollower's own default. With multiprocessing on, its
    constructor hands back a different object that keeps the real tracker -- and the fitted
    calibration -- in a subprocess, which puts both out of reach of `save`/`load` below. This
    is already a dedicated process whose only other job is to sleep, so there is nothing to
    gain by splitting it again.
    """
    try:
        from gazefollower import GazeFollower
    except ImportError:
        raise SystemExit(MISSING)
    return GazeFollower(use_multiprocessing=multiprocessing)


def save(follower, path: Path) -> bool:
    """Keep the fitted calibration, so the next run does not have to ask for it again."""
    model = getattr(follower, "calibration", None)
    weights = getattr(model, "weights", None)
    if weights is None or not getattr(model, "has_calibrated", False):
        return False
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weights=weights, t=time.time())
    print(f"[gaze] calibration saved to {path}", flush=True)
    return True


def load(follower, path: Path) -> bool:
    """Put a saved calibration back, if there is one this tracker can still use.

    Anything wrong with the file -- missing, corrupt, saved by a version whose features were
    a different width -- means calibrating again, which is a half-minute inconvenience and
    not an error worth stopping for. A *silently wrong* one would be far worse, so the shape
    is checked against the model rather than trusted.
    """
    model = getattr(follower, "calibration", None)
    if model is None or not path.exists():
        return False
    try:
        import numpy as np

        with np.load(path) as saved:
            weights, when = saved["weights"], float(saved["t"])
        if weights.ndim != 2 or weights.shape[1] != 2:
            raise ValueError(f"weights are {weights.shape}, expected (features + 1, 2)")
    except Exception as unusable:
        print(f"[gaze] ignoring {path}: {unusable}", flush=True)
        return False

    model.weights, model.has_calibrated = weights, True
    age = time.time() - when
    warning = (f" -- {int(age // 3600)}h old, and a calibration does not survive moving; "
               f"run --calibrate if the dot looks off") if age > STALE_CALIBRATION else ""
    fitted = time.strftime("%H:%M", time.localtime(when))
    print(f"[gaze] reusing the calibration fitted at {fitted}{warning}", flush=True)
    return True


def stream(follower, publisher: gaze.GazePublisher, hz: float, check: bool) -> None:
    """Publish looks until interrupted.

    Polled rather than subscribed because GazeFollower samples on its own thread and hands
    back the latest whenever asked, which is the same shape as the channel it feeds: one
    slot, newest wins, and a reader that misses three samples has missed nothing.
    """
    period = 1.0 / hz if hz > 0 else 0.0
    samples = blind = 0
    began = started = last_report = time.monotonic()
    ever = False
    print(f"[gaze] streaming to {publisher.path} -- ctrl-c to stop", flush=True)
    while True:
        if not ever and time.monotonic() - began > 5.0:
            # The tracker raises this per frame rather than once, so what it looks like from
            # out here is simply nothing arriving. Say the useful thing instead.
            raise SystemExit(
                "[gaze] the tracker produced nothing in 5s. The usual cause is no "
                "calibration -- it is required before sampling, and GazeFollower does not "
                "keep one across runs unless this tool saved it. Run:\n"
                "    ./scripts/gaze.sh --calibrate")
        info = follower.get_gaze_info()
        if info is not None and info.status:
            x, y = info.filtered_gaze_coordinates[:2]
            publisher.offer(float(x), float(y), valid=True)
            samples += 1
            ever = True
        else:
            # Publishing the blink rather than going silent is the point: the reader can
            # tell "the tracker is running and cannot see you" from "no tracker", and it
            # keeps the file's mtime fresh so a held lock is not dropped as stale.
            publisher.offer(0.0, 0.0, valid=False)
            blind += 1

        now = time.monotonic()
        if check and now - last_report >= 1.0:
            total = samples + blind
            where = (f"({info.filtered_gaze_coordinates[0]:7.1f}, "
                     f"{info.filtered_gaze_coordinates[1]:7.1f})"
                     if info is not None and info.status else "  -- no eyes --  ")
            print(f"  {where}  {total / max(1e-9, now - started):5.1f} Hz  "
                  f"{100 * samples / max(1, total):3.0f}% tracked", flush=True)
            last_report = now
            started, samples, blind = now, 0, 0
        if period:
            time.sleep(period)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="print a line a second with where you are looking and how much "
                             "of the time the tracker can see you -- how you tell a bad "
                             "calibration from a bug in the game")
    parser.add_argument("--no-preview", dest="preview", action="store_false",
                        help="skip the camera preview (it is how you check you are framed "
                             "and lit well enough before spending a calibration on it)")
    parser.add_argument("--calibrate", action="store_true",
                        help="fit a new calibration even if a saved one exists -- what to do "
                             "after moving your chair, your laptop or the lighting")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION,
                        help=f"where the fitted calibration is kept between runs "
                             f"(default {DEFAULT_CALIBRATION})")
    parser.add_argument("--hz", type=float, default=60.0,
                        help="how often to poll the tracker (default 60; it samples at the "
                             "camera's own rate, so this only has to be faster than that)")
    parser.add_argument("--path", default=gaze.DEFAULT_PATH,
                        help=f"where to publish (default {gaze.DEFAULT_PATH}); the game must "
                             f"agree, so set $MCAGENTS_GAZE_PATH rather than this if you "
                             f"change it")
    parser.add_argument("--multiprocessing", action="store_true",
                        help="run the tracker in a subprocess, GazeFollower's own default. "
                             "Its calibration then cannot be saved or reloaded, so this is "
                             "off here")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    follower = build(args.multiprocessing)
    try:
        calibrated = not args.calibrate and load(follower, args.calibration)
        if not calibrated:
            if args.preview:
                print("[gaze] camera preview -- sit how you will be sitting, check your face "
                      "is lit and in frame, then close it to go on", flush=True)
                follower.preview()
            print("[gaze] calibrating -- look at each dot and click it", flush=True)
            follower.calibrate()
            if not save(follower, args.calibration):
                print("[gaze] could not save the calibration; this run still has it",
                      flush=True)
        follower.start_sampling()
        stream(follower, gaze.GazePublisher(args.path), args.hz, args.check)
    except KeyboardInterrupt:
        print("\n[gaze] stopped", flush=True)
    finally:
        try:
            follower.stop_sampling()
        except Exception:
            pass
        follower.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
