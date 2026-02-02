from datetime import date
from pathlib import Path
from typing import Callable

from phylogenie.msa import MSA, Sequence


def load_fasta(
    fasta_file: str | Path,
    extract_time_from_id: Callable[[str], float | date] | None = None,
) -> MSA:
    """
    Load a FASTA file into an MSA object.

    Optionally extracts sampling times from sequence identifiers. If no custom
    extractor is provided, the loader attempts to parse the last '|' separated
    segment as a float or ISO date.

    Parameters
    -----------
    fasta_file : str | Path
        Path to the FASTA file to read.
    extract_time_from_id : Callable[[str], float | date] | None, optional
        Function that extracts a sampling time from a FASTA record ID.

    Returns
    --------
    MSA
        The loaded multiple sequence alignment.
    """
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
                last_metadata = id.split("|")[-1]
                try:
                    time = float(last_metadata)
                except ValueError:
                    try:
                        time = date.fromisoformat(last_metadata)
                    except ValueError:
                        pass
            chars = next(f).strip()
            sequences.append(Sequence(id, chars, time))
    return MSA(sequences)


def dump_fasta(msa: MSA | list[Sequence], fasta_file: str | Path):
    """
    Write an MSA or list of sequences to a FASTA file.

    Parameters
    -----------
    msa : MSA | list[Sequence]
        The alignment or raw sequences to write.
    fasta_file : str | Path
        Output file path.
    """
    with open(fasta_file, "w") as f:
        sequences = msa.sequences if isinstance(msa, MSA) else msa
        for seq in sequences:
            f.write(f">{seq.id}\n")
            f.write(f"{seq.chars}\n")
