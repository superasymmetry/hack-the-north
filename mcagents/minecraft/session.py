"""Booting a Minecraft the two controllers can share.

Everything between "python starts" and "the agent sees its first frame" is the same for both
of them: check the engine is downloaded, attach to a held instance if one was asked for,
build the sim, point it at the cached world, optionally lay the city over it, reset.

    with Session(EnvConfig.from_env()) as session:
        sim = session.open(callbacks=[PrevActionCallback()], obs_size=(224, 224))
        ...
"""
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from mcagents.minecraft import instance
from mcagents.minecraft.world import DEFAULT_BIOME, DEFAULT_WORLD, WorldCache, baked_city_path


def default_world() -> Path:
    """Chiang Tung when it has been put in worlds/, the plains cache otherwise."""
    return DEFAULT_WORLD if DEFAULT_WORLD.exists() else WorldCache().path


@dataclass
class EnvConfig:
    """How to get a Minecraft: a held instance or a fresh one, cached terrain, city or not."""

    #: Port of a `scripts/mc_server.sh` holder to attach to. None launches Minecraft in-process.
    mc_port: Optional[int] = None
    #: Terrain to load instead of generating (~13s a reset). None generates every time.
    world: Optional[Path] = None
    #: Put the agent in a city. Satisfied by loading a baked city world if one has been
    #: built, and otherwise by `CityCallback`, which costs ~10s of every reset.
    city: bool = False
    #: Whether `world` was named by the caller. An explicit choice is never second-guessed.
    explicit_world: bool = False

    @classmethod
    def from_env(cls, city_default: bool = False) -> "EnvConfig":
        world = os.environ.get("MC_WORLD")
        return cls(
            mc_port=int(os.environ["MC_PORT"]) if os.environ.get("MC_PORT") else None,
            world=Path(world) if world else default_world(),
            city=os.environ.get("CITY", "1" if city_default else "0") != "0",
            explicit_world=bool(world),
        )


class Session(AbstractContextManager):
    """Owns the sim for the length of a run, and closes it however the run ends."""

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.sim = None
        self.cache: Optional[WorldCache] = self._resolve_world()

    def _resolve_world(self) -> Optional[WorldCache]:
        """Pick the world, preferring a city that is already terrain over one built at reset.

        `city=True` asks to be *in* a city, not for a particular way of getting one. If
        `mcagents.minecraft.bake` has already put that same city into a world zip -- same
        seed, same layout, same spawn block -- loading it is the identical result for about
        10s less of every reset, so take it and leave the callback off.

        An explicitly named MC_WORLD is never second-guessed: asking for a specific world
        and silently getting a different one would be the worse surprise.
        """
        cache = WorldCache.from_path(self.config.world)
        if cache is None or not self.config.city:
            return cache

        if cache.is_baked_city:
            # The city is already in this terrain. Building a second one over it would take
            # ~10s of every reset to arrive at the same blocks it starts from.
            self.config.city = False
            return cache

        if self.config.explicit_world:
            return cache

        baked = WorldCache.from_path(baked_city_path(cache.biome, cache.directory))
        if not baked.exists:
            print(f"Building the city at every reset (~10s). Bake it into a world once and "
                  f"this becomes free:\n  bash scripts/bake_city.sh")
            return cache

        print(f"Using the baked city world {baked.path.name} instead of building one at "
              f"every reset.")
        self.config.city = False
        return baked

    @property
    def spawn_biome(self) -> str:
        """Ignored while a cached world is loaded (the zip *is* the seed), but right otherwise."""
        return self.cache.biome if self.cache else DEFAULT_BIOME

    def callbacks(self, *extra) -> List[Any]:
        """The requested callbacks with the city in front, where it has to be.

        CityCallback steps the env a few hundred times, which would leave anything
        PrevActionCallback seeded before it stale.
        """
        callbacks = list(extra)
        if self.config.city:
            from mcagents.minecraft.city import CityCallback
            callbacks.insert(0, CityCallback())
        return callbacks

    def open(self, callbacks: Optional[List[Any]] = None, **sim_kwargs):
        """Build the sim, load the cached world into it, and reset. Returns the live sim."""
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.entry import check_engine

        check_engine(skip_confirmation=True)

        if self.config.mc_port is not None:
            instance.attach(self.config.mc_port)
            print(f"Reusing the Minecraft instance on port {self.config.mc_port}.")

        sim_kwargs.setdefault("preferred_spawn_biome", self.spawn_biome)
        self.sim = MinecraftSim(callbacks=self.callbacks(*(callbacks or [])), **sim_kwargs)

        if self.cache and self.cache.exists:
            self.cache.apply(self.sim)
            print(f"Loading the cached world from {self.cache.path}.")
        elif self.cache:
            print(f"{self.cache.path} not found -- generating a fresh world, which is ~13s "
                  f"slower. Build the cache once with: bash scripts/build_world.sh")

        self.sim.reset()
        return self.sim

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def __exit__(self, *exc_info) -> None:
        self.close()
