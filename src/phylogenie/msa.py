from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass
class Sequence:
    """Represent a single aligned sequence with optional sampling time."""

    id: str
    chars: str
    time: float | date | None = None

    def __len__(self) -> int:
        return len(self.chars)


class MSA:
    """
    Store and validate a multiple sequence alignment.

    The alignment ensures that all sequences have identical length and provides
    convenient accessors for IDs, times, and alignment shape.
    """

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
    def ids(self) -> list[str]:
        """Return the sequence identifiers."""
        return [sequence.id for sequence in self.sequences]

    @property
    def times(self) -> list[float | date]:
        """Return the sequence sampling times."""
        times: list[float | date] = []
        for sequence in self:
            if sequence.time is None:
                raise ValueError(f"Time is not set for sequence {sequence.id}.")
            times.append(sequence.time)
        return times

    @property
    def alignment(self) -> list[list[str]]:
        """Return the alignment as a list of character lists."""
        return [list(sequence.chars) for sequence in self.sequences]

    @property
    def n_sequences(self) -> int:
        """Return the number of sequences in the alignment."""
        return len(self.sequences)

    @property
    def n_sites(self) -> int:
        """Return the number of aligned sites per sequence."""
        return len(self.alignment[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Return the alignment shape as (n_sequences, n_sites)."""
        return self.n_sequences, self.n_sites

    def count_informative_sites(self) -> int:
        """
        Count phylogenetically informative sites in the alignment.

        A site is informative if it has at least two character states each
        appearing in at least two sequences.
        """
        n_informative_sites = 0
        for column in np.array(self.alignment).T:
            column: np.typing.NDArray[np.str_]
            _, char_counts = np.unique(column, return_counts=True)
            is_informative_char = char_counts >= 2
            if (is_informative_char).sum() >= 2:
                n_informative_sites += 1
        return n_informative_sites

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
