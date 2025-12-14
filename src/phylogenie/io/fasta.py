from pathlib import Path
from typing import Callable

from phylogenie.msa import MSA, Sequence


def load_fasta(
    fasta_file: str | Path, extract_time_from_id: Callable[[str], float] | None = None
) -> MSA:
    sequences: list[Sequence] = []
    with open(fasta_file, "r") as f:
        for line in f:
            if not line.startswith(">"):
                raise ValueError(f"Invalid FASTA format: expected '>', got '{line[0]}'")
            id = line[1:].strip()
            time = None
            if extract_time_from_id is not None:
                time = extract_time_from_id(id)
            elif "|" in id:
                try:
                    time = float(id.split("|")[-1])
                except ValueError:
                    pass
            chars = next(f).strip()
            sequences.append(Sequence(id, chars, time))
    return MSA(sequences)
