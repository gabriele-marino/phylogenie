from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from typing import Annotated, Iterable, Literal

from pydantic import BaseModel, Field


class AlphabetType(str, Enum):
    GENERIC = "Generic"
    AMINO_ACIDS = "AminoAcids"
    NUCLEOTIDES = "Nucleotides"


TokenizedSequence = tuple[int, ...]


def _char_to_id(alphabet: str) -> dict[str, int]:
    return {char: i for i, char in enumerate(alphabet)}


class BaseAlphabet(ABC, BaseModel):
    type: AlphabetType

    @cached_property
    @abstractmethod
    def char_to_id(self) -> dict[str, int]: ...

    def tokenize_one(self, sequence: str) -> TokenizedSequence:
        return tuple(self.char_to_id[char] for char in sequence)

    def tokenize(
        self,
        sequences: Iterable[str],
    ) -> tuple[TokenizedSequence, ...]:
        return tuple(self.tokenize_one(sequence) for sequence in sequences)

    def __call__(
        self,
        sequences: Iterable[str],
    ) -> tuple[TokenizedSequence, ...]:
        return self.tokenize(sequences)

    def __len__(self) -> int:
        return len(self.char_to_id)


class GenericAlphabet(BaseAlphabet):
    type: Literal[AlphabetType.GENERIC] = AlphabetType.GENERIC
    alphabet: str

    @cached_property
    def char_to_id(self) -> dict[str, int]:
        return _char_to_id(self.alphabet)


class AminoAcidsAlphabet(BaseAlphabet):
    type: Literal[AlphabetType.AMINO_ACIDS] = AlphabetType.AMINO_ACIDS
    missing_char: str | None = None
    unidentified_char: str | None = None

    @cached_property
    def char_to_id(self) -> dict[str, int]:
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        if self.missing_char is not None:
            amino_acids += self.missing_char
        if self.unidentified_char is not None:
            amino_acids += self.unidentified_char
        return _char_to_id(amino_acids)


class NucleotidesAlphabet(BaseAlphabet):
    type: Literal[AlphabetType.NUCLEOTIDES] = AlphabetType.NUCLEOTIDES
    missing_char: str | None = None
    unidentified_char: str | None = None

    @cached_property
    def char_to_id(self) -> dict[str, int]:
        nucleotides = "ACGT"
        if self.missing_char is not None:
            nucleotides += self.missing_char
        if self.unidentified_char is not None:
            nucleotides += self.unidentified_char
        return _char_to_id(nucleotides)


Alphabet = Annotated[
    GenericAlphabet | AminoAcidsAlphabet | NucleotidesAlphabet,
    Field(discriminator="type"),
]
