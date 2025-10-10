from typing import Optional, Dict, List
import torch


class SeedGenerator:
    """
    Reproducible multi-device seed manager for PyTorch.

    Provides per-device `torch.Generator` instances and deterministic
    split seeds for reproducible sampling and model initialization
    across devices.
    """

    def __init__(self, seed: Optional[int] = None, devices: Optional[List[str]] = None):
        """
        Parameters
        ----------
        seed : int, optional
            Base seed for all RNGs. Randomized via torch.initial_seed() if None.
        devices : list of str, optional
            Devices to initialize (e.g., ['cpu', 'cuda:0']). Defaults to ['cpu'].
        """
        self._base_seed: int = int(seed if seed is not None else torch.initial_seed())
        self._devices: List[str] = devices or ["cpu"]
        self._generators: Dict[str, torch.Generator] = {}
        self._last_split_seeds: Dict[str, List[int]] = {}
        self._last_split_gens: Dict[str, List[torch.Generator]] = {}
        self._init_generators()

    def _init_generators(self):
        """Initialize or reset per-device generators."""
        for dev in self._devices:
            gen = torch.Generator(device=dev).manual_seed(self._base_seed)
            self._generators[dev] = gen
            self._last_split_seeds[dev] = []
            self._last_split_gens[dev] = []

    def add_device(self, device: str):
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
        Create `n` reproducible generators derived from the parent generator.

        Parameters
        ----------
        n : int
            Number of generators to create.
        device : str, default='cpu'
            Device identifier.

        Returns
        -------
        List[torch.Generator]
            List of reproducible generators for the device.
        """
        if device not in self._generators:
            raise ValueError(f"No generator found for device '{device}'")

        parent_gen = self._generators[device]
        # deterministic split seeds
        seeds = torch.randint(0, 2**63, (n,), dtype=torch.int64, generator=parent_gen)
        generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]

        self._last_split_seeds[device] = seeds.tolist()
        self._last_split_gens[device] = generators
        return generators

    split_generators = split  # alias for clarity

    def reseed(self, seed: Optional[int] = None):
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
            raise ValueError(f"No generator found for device '{device}'")
        return self._generators[device]

    @property
    def seed(self) -> int:
        """Return the current base seed."""
        return self._base_seed

    @seed.setter
    def seed(self, value: int):
        """Reset all device RNGs to a new base seed."""
        self.reseed(value)

    def last_split(self, device: str = "cpu") -> List[int]:
        """
        Return the most recent split seeds for a device.

        Parameters
        ----------
        device : str, default='cpu'
            Device identifier.

        Returns
        -------
        List[int]
            List of last split seeds.
        """
        return self._last_split_seeds.get(device, [])

    def last_generators(self, device: str = "cpu") -> List[torch.Generator]:
        """
        Return the most recent split generators for a device.

        Parameters
        ----------
        device : str, default='cpu'

        Returns
        -------
        List[torch.Generator]
        """
        return self._last_split_gens.get(device, [])

    def __call__(self) -> int:
        """Return the current base seed (functional usage)."""
        return self._base_seed

    def __repr__(self) -> str:
        devices = ", ".join(self._devices)
        return f"SeedGenerator(seed={self._base_seed}, devices=[{devices}])"
