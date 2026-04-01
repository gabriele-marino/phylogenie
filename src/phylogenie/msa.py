from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date

import numpy as np

SamplingTime = float | date


@dataclass
class Sequence:
    """Single aligned sequence with optional sampling time."""

    id: str
    chars: str
    time: SamplingTime | None = None

    def __len__(self) -> int:
        return len(self.chars)


class MSA:
    """Multiple sequence alignment (MSA) as a collection of sequences."""

    def __init__(self, sequences: Iterable[Sequence]):
        """Initialize the MSA with a collection of sequences."""
        self._sequences = list(sequences)
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) > 1:
            raise ValueError(
                f"All sequences in the alignment must have the same length (got lengths: {lengths})"
            )

    @property
    def sequences(self) -> tuple[Sequence, ...]:
        """Return the alignment sequences."""
        return tuple(self._sequences)

    @property
    def alignment(self) -> np.typing.NDArray[np.str_]:
        """Return the alignment as a 2D NumPy array of characters."""
        return np.array([list(sequence.chars) for sequence in self.sequences])

    @property
    def n_sequences(self) -> int:
        """Return the number of sequences in the alignment."""
        return len(self.sequences)

    @property
    def n_sites(self) -> int:
        """Return the number of aligned sites per sequence."""
        return len(self.sequences[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Return the alignment shape as (n_sequences, n_sites)."""
        return self.n_sequences, self.n_sites

    def count_unique_sequences(self) -> int:
        """Count the number of unique sequences in the alignment."""
        return len(np.unique(self.alignment, axis=0))

    def __len__(self) -> int:
        """Return the number of sequences in the alignment."""
        return self.n_sequences

    def __getitem__(self, item: int) -> Sequence:
        """Return the sequence at the specified index."""
        return self.sequences[item]

    def __iter__(self) -> Iterator[Sequence]:
        """Iterate over the sequences in the alignment."""
        return iter(self.sequences)
