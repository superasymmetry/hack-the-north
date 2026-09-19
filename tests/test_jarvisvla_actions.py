"""Check JarvisVLA's action decoder against the real action space.

No server and no Minecraft: this is pure token arithmetic, and every mistake it can make
looks, from the outside, like the model being bad at Minecraft.

    python tests/test_jarvisvla_actions.py
"""
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcagents.agents.jarvisvla_actions import (ACT_BEGIN, ACT_END, BASES, N_CAMERA_BINS,
                                               TOKEN_GROUPS, decode_actions, null_action_text,
                                               token)


def env_action(text: str) -> Dict[str, Any]:
    """Decode a reply and push it all the way through to a human-readable env action."""
    from minestudio.simulator.entry import CameraConfig
    from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
    from minestudio.utils.vpt_lib.actions import ActionTransformer

    config = CameraConfig(camera_binsize=1, camera_maxval=10, camera_mu=20,
                          camera_quantization_scheme="mu_law")
    assert config.n_camera_bins == N_CAMERA_BINS, config.n_camera_bins
    mapper = CameraHierarchicalMapping(n_camera_bins=config.n_camera_bins)
    transformer = ActionTransformer(**config.action_transformer_kwargs)

    action = decode_actions(text)[0]
    return transformer.policy2env(mapper.to_factored(dict(action)))


def pressed(action: Dict[str, Any]) -> Dict[str, Any]:
    """Only what is actually held down, so a case can state its expectation in one line."""
    values = {key: (np.asarray(value).reshape(-1).tolist() if key == "camera"
                    else int(np.asarray(value).reshape(())))
              for key, value in action.items()}
    return {key: value for key, value in values.items()
            if (value != [0.0, 0.0] if key == "camera" else value != 0)}


def tokens(*numbers: int) -> str:
    return "".join(token(n) for n in (ACT_BEGIN,) + numbers + (ACT_END,))


CASES = [
    ("null action", null_action_text(), {}),
    ("attack", tokens(204, 219, 240), {"attack": 1}),
    # Index 0 of every group means "not pressed", so forward is 191 and jump is 206 -- the
    # first token of each group (190, 205) is the no-op and must decode to nothing.
    ("forward + jump", tokens(191, 206, 219, 240), {"forward": 1, "jump": 1}),
    ("group no-ops", tokens(190, 205, 219, 240), {}),
    ("hotbar.1", tokens(181, 219, 240), {"hotbar.1": 1}),
    ("inventory", tokens(177, 219, 240), {"inventory": 1}),
    # 229 is the top pitch bin and 230 the bottom yaw bin: full deflection both ways.
    ("camera extremes", tokens(229, 230), {"camera": [10.0, -10.0]}),
    ("chatty preamble", "Sure! " + tokens(204, 219, 240), {"attack": 1}),
    ("no action at all", "I am not sure.", {}),
]


def main() -> int:
    failures = 0
    for label, text, expected in CASES:
        got = pressed(env_action(text))
        ok = got == expected
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<20} -> {got or '{}'}"
              + ("" if ok else f"   expected {expected or '{}'}"))

    # Two actions in one reply must come back as two, in order -- that is what makes a
    # chunk_len above 1 mean anything.
    chunk = decode_actions(tokens(204, 219, 240) + tokens(190, 219, 240))
    assert len(chunk) == 2, chunk
    print(f"  ok   action chunking      -> {len(chunk)} actions in one reply")

    print(f"\n{'decoder OK' if not failures else f'{failures} FAILURES'}: "
          f"{len(TOKEN_GROUPS)} groups, bases {BASES}, {N_CAMERA_BINS} camera bins")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
