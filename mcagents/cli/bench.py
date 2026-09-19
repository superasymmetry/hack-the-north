"""Measure raw environment throughput: steps per second with no policy in the loop.

    python -m mcagents.cli.bench --steps 500

The number to compare a controller's FPS against -- if the env alone runs at 30 steps/s, no
policy on top of it will do better.
"""
import argparse
import time

from minestudio.simulator import MinecraftSim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=500)
    steps = parser.parse_args().steps

    sim = MinecraftSim(action_type="env")
    sim.reset()
    started = time.time()
    for _ in range(steps):
        sim.step(sim.action_space.sample())
    elapsed = time.time() - started
    sim.close()

    print(f"{steps} steps in {elapsed:.2f}s -> {steps / elapsed:.2f} steps/sec")


if __name__ == "__main__":
    main()
