"""Play Minecraft yourself in the simulator -- the fastest way to see what an agent sees.

    bash scripts/play.sh

Needs the GUI patches from scripts/setup/ and a DISPLAY.
"""
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import PlayCallback


def main() -> None:
    sim = MinecraftSim(obs_size=(224, 224), action_type="env",
                       callbacks=[PlayCallback(agent_generator=None)])
    sim.reset()
    terminated = False
    try:
        while not terminated:
            _, _, terminated, _, _ = sim.step(None)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
