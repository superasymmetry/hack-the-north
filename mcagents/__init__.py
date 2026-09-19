"""Goal-conditioned Minecraft agents on top of MineStudio.

Two controllers share one environment stack:

    mcagents.agents.rocket2     a point + an interaction type -> low-level control
    mcagents.agents.jarvisvla   an English sentence -> low-level control (remote 7B VLM)

Nothing heavy is imported here: `mcagents.agents.rocket2` pulls in torch and the vendored
ROCKET-2 policy, and importing this package should not cost that.
"""

__all__ = ["agents", "minecraft", "perception"]
