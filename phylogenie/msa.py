from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass
class Sequence:
    id: str
    chars: str
    time: float | None = None


class MSA:
    def __init__(self, sequences: list[Sequence]):
        self.sequences = sequences
        lengths = {len(sequence.chars) for sequence in sequences}
        if len(lengths) > 1:
            raise ValueError(
                f"All sequences in the alignment must have the same length (got lengths: {lengths})"
            )

    @property
    def ids(self) -> list[str]:
        return [sequence.id for sequence in self.sequences]

    @property
    def times(self) -> list[float]:
        times: list[float] = []
        for sequence in self:
            if sequence.time is None:
                raise ValueError(f"Time is not set for sequence {sequence.id}.")
            times.append(sequence.time)
        return times

    @property
    def alignment(self) -> list[list[str]]:
        return [list(sequence.chars) for sequence in self.sequences]

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    @property
    def n_sites(self) -> int:
        return len(self.alignment[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_sequences, self.n_sites

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, item: int) -> Sequence:
        return self.sequences[item]

    def __iter__(self) -> Iterator[Sequence]:
        return iter(self.sequences)

    def count_informative_sites(self) -> int:
        n_informative_sites = 0
        for column in np.array(self.alignment).T:
            column: np.typing.NDArray[np.str_]
            _, char_counts = np.unique(column, return_counts=True)
            is_informative_char = char_counts >= 2
            if (is_informative_char).sum() >= 2:
                n_informative_sites += 1
        return n_informative_sites

    def count_unique_sequences(self) -> int:
        return len(np.unique(self.alignment, axis=0))
