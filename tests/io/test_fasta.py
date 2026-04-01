from pathlib import Path

import pytest

from phylogenie.io.fasta import dump_fasta, load_fasta
from phylogenie.msa import MSA, Sequence


def test_load_fasta_basic(tmp_path: Path):
    fasta = ">seq1\nAC\nGT\n>seq2\nA-GT\n"
    path = tmp_path / "test.fasta"
    path.write_text(fasta, encoding="utf-8")

    msa = load_fasta(path)

    assert [seq.id for seq in msa] == ["seq1", "seq2"]
    assert [seq.chars for seq in msa] == ["ACGT", "A-GT"]


def test_load_fasta_extract_time(tmp_path: Path):
    fasta = ">seq1|1.5\nACGT\n>seq2|2.0\nACGT\n"
    path = tmp_path / "test.fasta"
    path.write_text(fasta, encoding="utf-8")

    msa = load_fasta(path, extract_time_from_id=lambda s: float(s.split("|")[1]))

    assert [seq.time for seq in msa] == [1.5, 2.0]


def test_load_fasta_invalid_header(tmp_path: Path):
    path = tmp_path / "bad.fasta"
    path.write_text("seq1\nACGT\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_fasta(path)


def test_dump_fasta(tmp_path: Path):
    msa = MSA([Sequence("seq1", "ACGT"), Sequence("seq2", "A-GT")])
    path = tmp_path / "out.fasta"

    dump_fasta(msa, path)

    assert path.read_text(encoding="utf-8") == ">seq1\nACGT\n>seq2\nA-GT\n"
