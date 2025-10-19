from typing import Optional, Dict, List
import torch


class SeedGenerator:
    """
    Deterministic multi-device seed manager for PyTorch.

    Features:
        - Reproducible per-device `torch.Generator` instances.
        - Deterministic split generators for parallel/distributed computations.
        - Easy reseeding and generator retrieval across devices.
        - Lazy device addition.
    """

    def __init__(self, seed: Optional[int] = None, devices: Optional[List[str]] = None):
        self._base_seed: int = int(seed if seed is not None else torch.initial_seed())
        default_devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
        self._devices: List[str] = devices or default_devices

        self._generators: Dict[str, torch.Generator] = {}
        self._last_split_seeds: Dict[str, List[int]] = {}
        self._last_split_gens: Dict[str, List[torch.Generator]] = {}

        self._init_generators()

    def _init_generators(self) -> None:
        for dev in self._devices:
            self._generators[dev] = torch.Generator(device=dev).manual_seed(self._base_seed)
            self._last_split_seeds[dev] = []
            self._last_split_gens[dev] = []

    def add_device(self, device: str) -> None:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device).manual_seed(self._base_seed)
            self._last_split_seeds[device] = []
            self._last_split_gens[device] = []
            if device not in self._devices:
                self._devices.append(device)

    def split(self, n: int, device: str = "cpu") -> List[torch.Generator]:
        if device not in self._generators:
            self.add_device(device)
        if n <= 0:
            raise ValueError("Number of splits must be positive")

        parent_gen = self._generators[device]
        seeds = torch.randint(0, 2**63, (n,), dtype=torch.int64, generator=parent_gen)
        generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]

        self._last_split_seeds[device] = seeds.tolist()
        self._last_split_gens[device] = generators
        return generators

    split_generators = split  # alias

    def reseed(self, seed: Optional[int] = None) -> None:
        self._base_seed = int(seed if seed is not None else torch.initial_seed())
        self._init_generators()

    def get(self, device: str = "cpu") -> torch.Generator:
        if device not in self._generators:
            self.add_device(device)
        return self._generators[device]

    @property
    def seed(self) -> int:
        return self._base_seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.reseed(value)

    def last_split(self, device: str = "cpu") -> List[int]:
        return self._last_split_seeds.get(device, [])

    def last_generators(self, device: str = "cpu") -> List[torch.Generator]:
        return self._last_split_gens.get(device, [])

    def reproducible(self, device: str = "cpu") -> bool:
        seeds = self._last_split_seeds.get(device, [])
        return len(seeds) == len(set(seeds))

    def __call__(self) -> int:
        return self._base_seed

    def __repr__(self) -> str:
        devices = ", ".join(self._devices)
        return f"SeedGenerator(base_seed={self._base_seed}, devices=[{devices}])"
