"""Track an index finger and publish it while thumb and index are pinched.

    ./scripts/hand.sh
    ./scripts/hand.sh --camera 0 --check

The camera preview uses the same MediaPipe skeleton and pinch test as ``tracker.py``.
The fingertip is converted from the mirrored camera image to physical screen pixels and
published through the existing pointing channel, so Rocket2 can use it without knowing
whether the point came from eyes or a hand.
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import time
from typing import Optional, Sequence, Tuple

from mcagents import gaze


def screen_size() -> Tuple[int, int]:
    """Return the X11 screen size, with an environment override for unusual setups."""
    override = os.environ.get("MCAGENTS_SCREEN_SIZE", "")
    if "x" in override.lower():
        width, height = override.lower().split("x", 1)
        return int(width), int(height)
    if shutil.which("xdpyinfo"):
        try:
            output = subprocess.check_output(["xdpyinfo"], text=True, timeout=2)
            match = re.search(r"dimensions:\s+(\d+)x(\d+) pixels", output)
            if match:
                return int(match.group(1)), int(match.group(2))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return 1920, 1080


def pinch_distance(index, thumb) -> float:
    return math.hypot(index.x - thumb.x, index.y - thumb.y)


def screen_point(index, width: int, height: int) -> Tuple[float, float]:
    """Map the mirrored camera landmark to screen pixels."""
    return float(index.x * width), float(index.y * height)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument("--path", default=gaze.DEFAULT_PATH,
                        help=f"pointing channel path (default {gaze.DEFAULT_PATH})")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="normalized thumb-index distance for a pinch")
    parser.add_argument("--check", action="store_true",
                        help="print the published fingertip once per second")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        import cv2
        import mediapipe as mp
    except ImportError as missing:
        raise SystemExit("hand tracking needs opencv-python and mediapipe; run "
                         "bash scripts/setup/install_gaze.sh") from missing

    hands_api = mp.solutions.hands
    draw = mp.solutions.drawing_utils
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    width, height = screen_size()
    publisher = gaze.GazePublisher(args.path)
    last_report = time.monotonic()
    try:
        with hands_api.Hands(max_num_hands=1, min_detection_confidence=0.5,
                             min_tracking_confidence=0.5) as hands:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pinch = False
                point = None
                landmarks_on_screen = None
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    landmarks = hand.landmark
                    index = landmarks[hands_api.HandLandmark.INDEX_FINGER_TIP]
                    thumb = landmarks[hands_api.HandLandmark.THUMB_TIP]
                    pinch = pinch_distance(index, thumb) < args.threshold
                    point = screen_point(index, width, height)
                    landmarks_on_screen = [screen_point(landmark, width, height)
                                           for landmark in landmarks]
                    draw.draw_landmarks(frame, hand, hands_api.HAND_CONNECTIONS)
                    frame_height, frame_width = frame.shape[:2]
                    cv2.circle(frame, (int(index.x * frame_width),
                                       int(index.y * frame_height)), 10, (0, 255, 0), -1)

                if pinch and point is not None:
                    publisher.offer(*point, valid=True, landmarks=landmarks_on_screen)
                else:
                    publisher.offer(0.0, 0.0, valid=False, landmarks=landmarks_on_screen)

                text = "PINCH!" if pinch else "OPEN"
                color = (0, 0, 255) if pinch else (0, 255, 0)
                cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2)
                if args.check and time.monotonic() - last_report >= 1.0:
                    where = (f"({point[0]:.0f}, {point[1]:.0f})" if pinch and point
                             else "-- no pinch --")
                    print(f"[hand] {where}", flush=True)
                    last_report = time.monotonic()
                cv2.imshow("Hand Tracker", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())