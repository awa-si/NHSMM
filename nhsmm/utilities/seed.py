from typing import Optional, Dict, List
import torch


class SeedGenerator:
    """
    Deterministic multi-device seed manager for PyTorch.

    Provides:
        - Reproducible per-device `torch.Generator` instances.
        - Deterministic split seeds for parallel or distributed sampling.
        - Easy reseeding and generator retrieval across devices.
    """

    def __init__(self, seed: Optional[int] = None, devices: Optional[List[str]] = None):
        """
        Parameters
        ----------
        seed : int, optional
            Base seed for all RNGs. Uses `torch.initial_seed()` if None.
        devices : list of str, optional
            Devices to initialize (e.g., ['cpu', 'cuda:0']).
            Defaults to ['cuda:0'] if CUDA is available, else ['cpu'].
        """
        default_devices = (
            ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
        )
        self._devices: List[str] = devices or default_devices
        self._base_seed: int = int(seed if seed is not None else torch.initial_seed())

        self._generators: Dict[str, torch.Generator] = {}
        self._last_split_seeds: Dict[str, List[int]] = {}
        self._last_split_gens: Dict[str, List[torch.Generator]] = {}

        self._init_generators()

    def _init_generators(self) -> None:
        """Initialize per-device generators from the base seed."""
        for dev in self._devices:
            gen = torch.Generator(device=dev).manual_seed(self._base_seed)
            self._generators[dev] = gen
            self._last_split_seeds[dev] = []
            self._last_split_gens[dev] = []

    def add_device(self, device: str) -> None:
        """Add a new device generator dynamically if not already present."""
        if device not in self._generators:
            gen = torch.Generator(device=device).manual_seed(self._base_seed)
            self._generators[device] = gen
            self._last_split_seeds[device] = []
            self._last_split_gens[device] = []
            if device not in self._devices:
                self._devices.append(device)

    def split(self, n: int, device: str = "cpu") -> List[torch.Generator]:
        """
        Create `n` deterministic generators derived from the parent generator.

        Parameters
        ----------
        n : int
            Number of generators to create.
        device : str, default='cpu'
            Device identifier.

        Returns
        -------
        List[torch.Generator]
            List of reproducible generators for the given device.
        """
        if device not in self._generators:
            raise ValueError(f"No generator found for device '{device}'")
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
        """
        Reseed all device generators with a new base seed.

        Parameters
        ----------
        seed : int, optional
            New base seed. If None, uses torch.initial_seed().
        """
        self._base_seed = int(seed if seed is not None else torch.initial_seed())
        self._init_generators()

    def get(self, device: str = "cpu") -> torch.Generator:
        """
        Retrieve the RNG for a given device.

        Parameters
        ----------
        device : str, default='cpu'
            Device identifier.

        Returns
        -------
        torch.Generator
            Generator for the requested device.
        """
        if device not in self._generators:
            self.add_device(device)
        return self._generators[device]

    @property
    def seed(self) -> int:
        """Return the current base seed."""
        return self._base_seed

    @seed.setter
    def seed(self, value: int) -> None:
        """Reset all device RNGs to a new base seed."""
        self.reseed(value)

    def last_split(self, device: str = "cpu") -> List[int]:
        """Return the most recent split seeds for a given device."""
        return self._last_split_seeds.get(device, [])

    def last_generators(self, device: str = "cpu") -> List[torch.Generator]:
        """Return the most recent split generators for a given device."""
        return self._last_split_gens.get(device, [])

    def reproducible(self, device: str = "cpu") -> bool:
        """
        Check if last split generators reproduce the same seeds.

        Returns
        -------
        bool
            True if all last split generators produce the same sequence.
        """
        seeds = self._last_split_seeds.get(device, [])
        return len(seeds) == len(set(seeds))

    def __call__(self) -> int:
        """Return the current base seed (functional usage)."""
        return self._base_seed

    def __repr__(self) -> str:
        devices = ", ".join(self._devices)
        return f"SeedGenerator(base_seed={self._base_seed}, devices=[{devices}])"
