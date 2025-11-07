import torch
from typing import Optional, Dict, List


class SeedGenerator:
    """
    Deterministic multi-device seed manager for PyTorch.

    Features:
        - Reproducible per-device `torch.Generator` instances.
        - Deterministic split generators for parallel/distributed computations.
        - Easy reseeding and generator retrieval across devices.
        - Lazy device registration with consistent reproducibility.
    """

    def __init__(self, seed: Optional[int] = None, devices: Optional[List[str]] = None):
        self._base_seed: int = int(seed if seed is not None else torch.seed())
        default_devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
        self._devices: List[str] = list(devices or default_devices)

        self._generators: Dict[str, torch.Generator] = {}
        self._last_split_seeds: Dict[str, List[int]] = {}
        self._last_split_gens: Dict[str, List[torch.Generator]] = {}

        self._init_generators()

    def _init_generators(self) -> None:
        """Initialize per-device generators using the base seed."""
        for dev in self._devices:
            gen = torch.Generator(device=dev)
            gen.manual_seed(self._base_seed)
            self._generators[dev] = gen
            self._last_split_seeds[dev] = []
            self._last_split_gens[dev] = []

    def add_device(self, device: str) -> None:
        """Register a new device and initialize its generator."""
        if device not in self._generators:
            gen = torch.Generator(device=device)
            gen.manual_seed(self._base_seed)
            self._generators[device] = gen
            self._last_split_seeds[device] = []
            self._last_split_gens[device] = []
            if device not in self._devices:
                self._devices.append(device)

    def split(self, n: int, device: str = "cpu") -> List[torch.Generator]:
        """
        Deterministically split the base generator on a device into `n` sub-generators.

        Args:
            n: Number of sub-generators to create.
            device: Device string (e.g., 'cpu', 'cuda:0').

        Returns:
            List of `torch.Generator` instances seeded independently.
        """
        if n <= 0:
            raise ValueError("Number of splits must be positive.")
        if device not in self._generators:
            self.add_device(device)

        parent_gen = self._generators[device]
        # Use uint64 range to avoid overflow ambiguity
        seeds = torch.randint(0, 2**63 - 1, (n,), dtype=torch.int64, generator=parent_gen, device="cpu")
        generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]

        self._last_split_seeds[device] = seeds.tolist()
        self._last_split_gens[device] = generators
        return generators

    split_generators = split  # alias

    def reseed(self, seed: Optional[int] = None) -> None:
        """Reseed all generators with a new base seed."""
        self._base_seed = int(seed if seed is not None else torch.seed())
        self._init_generators()

    def get(self, device: str = "cpu") -> torch.Generator:
        """Retrieve generator for the given device (auto-initialized if needed)."""
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
        """Return the most recent split seed list for a device."""
        return self._last_split_seeds.get(device, [])

    def last_generators(self, device: str = "cpu") -> List[torch.Generator]:
        """Return the most recent split generator list for a device."""
        return self._last_split_gens.get(device, [])

    def reproducible(self, device: str = "cpu") -> bool:
        """Check that last split seeds are unique (no duplicates)."""
        seeds = self._last_split_seeds.get(device, [])
        return len(seeds) == len(set(seeds)) and len(seeds) > 0

    def __call__(self) -> int:
        """Return current base seed."""
        return self._base_seed

    def __repr__(self) -> str:
        devices = ", ".join(self._devices)
        return f"SeedGenerator(base_seed={self._base_seed}, devices=[{devices}])"
