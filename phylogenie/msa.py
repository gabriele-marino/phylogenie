from dataclasses import dataclass
from typing import Callable

import numpy as np
from Bio.SeqIO import parse

from phylogenie.utils.type_hints import Numeric


def default_extract_sequence_date(name: str) -> float:
    return float(name.split("|")[-1])


@dataclass
class Sequence:
    name: str
    date: float
    sequence: str


class MSA:
    def __init__(
        self,
        msa_file: str,
        extract_sequence_date: Callable[[str], float] = default_extract_sequence_date,
    ):
        Bio_sequences = tuple(parse(msa_file, "fasta"))
        self.names = tuple(sequence.name for sequence in Bio_sequences)
        self.dates = tuple(
            extract_sequence_date(sequence.name) for sequence in Bio_sequences
        )
        self.alignment = tuple(str(sequence.seq) for sequence in Bio_sequences)
        self.sequences = tuple(
            Sequence(name=name, date=date, sequence=sequence)
            for name, date, sequence in zip(self.names, self.dates, self.alignment)
        )
        if not any(
            len(sequence) != len(self.alignment[0]) for sequence in self.alignment
        ):
            raise ValueError(
                "All sequences in the alignment must have the same length."
            )
        self.n_sequences = len(self.sequences)

    def __len__(self) -> int:
        return self.n_sequences

    def count_informative_sites(
        self,
        min_number_of_char_occurrences: int = 2,
        min_number_of_chars: int = 2,
        relative: bool = False,
    ) -> Numeric:
        alignment = np.array(tuple(tuple(sequence) for sequence in self.alignment))
        _, n_sites = alignment.shape
        informative_sites = tuple(
            idx
            for idx, column in enumerate(alignment.T)
            if (
                np.unique(column, return_counts=True)[1]
                >= min_number_of_char_occurrences
            ).sum()
            >= min_number_of_chars
        )
        n_informative_sites = len(informative_sites)
        if relative:
            return n_informative_sites / n_sites
        return n_informative_sites

    def count_unique_sequences(self, relative: bool = False) -> Numeric:
        alignment = np.array(tuple(tuple(sequence) for sequence in self.alignment))
        n_unique_sequences = len(np.unique(alignment, axis=0))
        if relative:
            return n_unique_sequences / self.n_sequences
        return n_unique_sequences
