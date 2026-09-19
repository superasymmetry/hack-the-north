"""JarvisVLA's action vocabulary: reserved special tokens <-> VPT (buttons, camera) actions.

The model emits its action as a run of `<|reserved_special_token_N|>` tokens. This decoder
works on those *names* rather than re-tokenizing the reply, which is where upstream needs
`AutoTokenizer.from_pretrained(checkpoint)`: same answer, no 7B repo to pull onto a laptop.
The two are equivalent because the reserved tokens are contiguous in this checkpoint --
id = 151657 + N, verified against added_tokens.json on the Hub -- so the name carries N.

Two subtleties, both of which fail silently rather than raising:

* **Camera is hierarchical.** The two camera bins are only read when the camera meta-button
  is set in `buttons` (CameraHierarchicalMapping.to_factored zeroes them otherwise). The
  model never emits that bit; upstream reconstructs it at decode time from "the camera bins
  are not both centred", and so does `decode_actions` below.
* **The camera config is part of the checkpoint.** These tokens are 21-way, i.e.
  CameraConfig(binsize=1, maxval=10, mu=20). Under MinecraftSim's 11-bin default every
  camera token is read against the wrong table and the agent looks drunk.

    python -m mcagents.cli.jarvisvla --selftest      # decoder only: no server, no Minecraft
"""
import re
from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np

#: Reserved-token id = TOKEN_ID_BASE + N. The decoder works on N; this is what makes that
#: equivalent to upstream's tokenize-then-decode.
TOKEN_ID_BASE = 151657

#: Each action is bracketed by these two.
ACT_BEGIN, ACT_END = 178, 179

#: Reserved-token numbers per action group, transcribed from JarvisVLA's
#: inference/action_mapping.py:map_control_token. Order matters: it *is* the digit order of
#: the mixed-base number that becomes (buttons, camera).
TOKEN_GROUPS: List[List[int]] = [
    list(range(180, 190)),   # 0  hotbar.1 .. hotbar.9 (index 0 = none)
    [190, 191, 192],         # 1  none / forward / back
    [193, 194, 195],         # 2  none / left / right
    [196, 197, 198],         # 3  none / sprint / sneak
    [199, 200],              # 4  use
    [201, 202],              # 5  drop
    [203, 204],              # 6  attack
    [205, 206],              # 7  jump
    [207, 208],              # 8  camera meta-button
    [176, 177],              # 9  inventory -- out of sequence upstream, not a typo
    list(range(209, 230)),   # 10 camera pitch bin, 21 of them
    list(range(230, 251)),   # 11 camera yaw bin, 21 of them
]

#: Digit bases, one per group. The first nine multiply out to 8640, VPT's button space.
BASES = [len(group) for group in TOKEN_GROUPS]

#: VPT reserves the last button index for "inventory pressed" rather than giving it a digit.
INVENTORY_BUTTON = 8640

#: Group indices, named so the packing below reads as something other than arithmetic.
CAMERA_BUTTON_GROUP = 8
INVENTORY_GROUP = 9
PITCH_GROUP, YAW_GROUP = 10, 11

#: The camera bin meaning "no movement" -- the middle one of 21.
CAMERA_NULL_BIN = BASES[YAW_GROUP] // 2

#: What n_camera_bins the sim must be configured with for these tokens to mean anything.
N_CAMERA_BINS = BASES[YAW_GROUP]

TOKEN_RE = re.compile(r"<\|reserved_special_token_(\d+)\|>")

_TOKEN_TO_SLOT: Dict[int, tuple] = {
    number: (place, index)
    for place, group in enumerate(TOKEN_GROUPS)
    for index, number in enumerate(group)
}

AgentAction = "OrderedDict[str, np.ndarray]"


def token(number: int) -> str:
    return f"<|reserved_special_token_{number}|>"


def null_group() -> List[int]:
    """The group-digit action that does nothing: no buttons, both camera bins centred."""
    return [0] * (len(BASES) - 2) + [CAMERA_NULL_BIN, CAMERA_NULL_BIN]


def null_action_text() -> str:
    """The token string for "do nothing", used to fill history before the first response.

    Upstream builds this by encoding the null action; its encoder emits only the non-zero
    button groups plus *both* camera groups unconditionally, so for the null action that is
    just the two centred camera tokens between the tags.
    """
    return "".join(token(n) for n in (ACT_BEGIN,
                                      TOKEN_GROUPS[PITCH_GROUP][CAMERA_NULL_BIN],
                                      TOKEN_GROUPS[YAW_GROUP][CAMERA_NULL_BIN],
                                      ACT_END))


def group_to_action(group: Sequence[int]) -> AgentAction:
    """One group-digit action -> the {"buttons", "camera"} pair the agent action space wants.

    The button digits are a mixed-base number over groups 0-8; group 9 (inventory) is not a
    digit at all but a flag that replaces the whole button index; groups 10-11 are the camera
    bins, packed as pitch * 21 + yaw.
    """
    buttons = 0
    for place in range(CAMERA_BUTTON_GROUP + 1):
        buttons = buttons * BASES[place] + group[place]
    if group[INVENTORY_GROUP]:
        buttons = INVENTORY_BUTTON
    camera = group[PITCH_GROUP] * BASES[YAW_GROUP] + group[YAW_GROUP]
    # Shape (1,) rather than a scalar: to_factored asserts on the trailing axis.
    return OrderedDict(buttons=np.array([buttons]), camera=np.array([camera]))


def decode_actions(text: str) -> List[AgentAction]:
    """The model's reply -> the agent-space actions it contains, in order.

    Anything that is not a recognised control token between a begin/end pair is ignored, so a
    chatty preamble or a truncated final action costs at most that action. A reply with no
    complete action decodes to a single null action, which is upstream's behaviour and the
    right one here: a dropped frame should stall the agent for one step, not crash a run.
    """
    numbers = [int(match.group(1)) for match in TOKEN_RE.finditer(text)]

    groups: List[List[int]] = []
    index = 0
    while index < len(numbers):
        try:
            begin = numbers.index(ACT_BEGIN, index)
            end = numbers.index(ACT_END, begin + 1)
        except ValueError:
            break

        group = null_group()
        for number in numbers[begin + 1:end]:
            slot = _TOKEN_TO_SLOT.get(number)
            if slot is not None:
                place, value = slot
                group[place] = value

        # The camera bins are dead weight unless the meta-button is set, and the model does
        # not emit it -- see the module docstring.
        if group[PITCH_GROUP:] != [CAMERA_NULL_BIN, CAMERA_NULL_BIN]:
            group[CAMERA_BUTTON_GROUP] = 1

        groups.append(group)
        index = end + 1

    return [group_to_action(group) for group in (groups or [null_group()])]


def describe_action(action: Dict[str, np.ndarray], sim) -> str:
    """A one-line, human-readable form of an agent-space action, for logs."""
    env_action = sim.agent_action_to_env_action({k: np.asarray(v).copy() for k, v in action.items()})
    pressed = [key for key, value in env_action.items()
               if key != "camera" and int(np.asarray(value).reshape(())) != 0]
    pitch, yaw = np.asarray(env_action["camera"]).reshape(2)
    parts = pressed or ["-"]
    if abs(pitch) > 1e-6 or abs(yaw) > 1e-6:
        parts = parts + [f"camera({pitch:+.2f},{yaw:+.2f})"]
    return " ".join(parts)
