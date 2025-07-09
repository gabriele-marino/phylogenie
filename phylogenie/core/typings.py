import phylogenie.typings as pgt

Data = dict[
    str,
    str
    | pgt.Scalar
    | list[pgt.Scalar]
    | list[list[pgt.Scalar]]
    | list[list[list[pgt.Scalar]]],
]
