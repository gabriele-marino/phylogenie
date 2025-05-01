from Bio.Seq import MutableSeq, Seq

class SeqRecord:
    name: str
    seq: Seq | MutableSeq
