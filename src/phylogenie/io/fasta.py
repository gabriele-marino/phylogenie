from pathlib import Path
from typing import Callable

from phylogenie.msa import MSA, SamplingTime, Sequence


def load_fasta(
    fasta_file: str | Path,
    extract_time_from_id: Callable[[str], SamplingTime] | None = None,
) -> MSA:
    """
    Load a FASTA file into an MSA object, optionally extracting sampling times from sequence IDs.
    """

    def _init_sequence(line: str) -> Sequence:
        new_id = line[1:].strip()
        time = None if extract_time_from_id is None else extract_time_from_id(new_id)
        return Sequence(id=new_id, chars="", time=time)

    sequences: list[Sequence] = []
    with open(fasta_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line.startswith(">"):
            raise ValueError(
                "FASTA file must start with a header line (starting with '>')."
            )
        current_sequence = _init_sequence(first_line)
        for line in f:
            if line.startswith(">"):
                sequences.append(current_sequence)
                current_sequence = _init_sequence(line)
            else:
                current_sequence.chars += line.strip()
    sequences.append(current_sequence)
    return MSA(sequences)


def dump_fasta(msa: MSA | list[Sequence], fasta_file: str | Path):
    """Write an MSA or list of sequences to a FASTA file."""
    with open(fasta_file, "w", encoding="utf-8") as f:
        sequences = msa.sequences if isinstance(msa, MSA) else msa
        for seq in sequences:
            f.write(f">{seq.id}\n")
            f.write(f"{seq.chars}\n")
