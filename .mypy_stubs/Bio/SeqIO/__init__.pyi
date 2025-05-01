from typing import Any, Iterator

from Bio.SeqRecord import SeqRecord

def parse(handle: str, format: str, alphabet: Any = None) -> Iterator[SeqRecord]: ...
