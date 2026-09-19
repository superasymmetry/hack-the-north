"""Check what a targeter finds for a query, on a still frame, before trusting it in a run.

    python -m mcagents.cli.locate frame.png "tree"
    MCAGENTS_TARGETER=vlm python -m mcagents.cli.locate frame.png "the nearest tree"

Writes <frame>.boxes.png with the boxes drawn on, best-first.
"""
import argparse
import os

import cv2

from mcagents.perception.detection import annotate, load_targeter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image")
    parser.add_argument("text")
    parser.add_argument("--targeter", choices=["owl", "vlm"])
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    targeter = load_targeter(args.targeter)
    found = targeter.detect(frame, args.text)
    if not found:
        print(f"no match for {args.text!r}")
    for detection in found:
        x0, y0, x1, y1 = detection.box
        print(f"  {detection.score:.3f}  box=({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})  "
              f"point=({detection.point[0]:.0f},{detection.point[1]:.0f})")

    out = f"{os.path.splitext(args.image)[0]}.boxes.png"
    cv2.imwrite(out, cv2.cvtColor(annotate(frame, found), cv2.COLOR_RGB2BGR))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
