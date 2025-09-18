import random
from dataclasses import dataclass
from typing import List

@dataclass
class WorldConfig:
    width: int
    height: int


@dataclass
class ItemSpec:
    image: str
    count: int

@dataclass
class Environment:
    seed: int
    world: WorldConfig
    obstacles: List[ItemSpec]
    sites: List[ItemSpec]


def build_environment(cfg: Environment):

    prng = random.Random(cfg.seed)
    env = {"obstacles": [], "sites": []} #dict that holds all of our sites and obstacles

    for spec in cfg.obstacles:
        for _ in range(spec.count):
            env["obstacles"].append({
                "image": spec.image,
                "x": prng.randint(0, cfg.world.width),
                "y": prng.randint(0, cfg.world.height),
            })

    for spec in cfg.sites:
        for _ in range(spec.count):
            env["sites"].append({
                "image": spec.image,
                "x": prng.randint(0, cfg.world.width),
                "y": prng.randint(0, cfg.world.height),
            })

    return env