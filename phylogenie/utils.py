from typing import overload

import phylogenie.typeguards as tg
import phylogenie.typings as pgt


@overload
def vectorify1D(x: pgt.OneOrMany[int]) -> pgt.IntVector1D: ...
@overload
def vectorify1D(x: pgt.OneOrManyScalars | None) -> pgt.Vector1D: ...
def vectorify1D(x: pgt.OneOrManyScalars | None) -> pgt.Vector1D:
    if x is None:
        return ()
    if isinstance(x, pgt.Scalar):
        return (x,)
    if tg.is_many_scalars(x):
        return tuple(x)
    raise TypeError(
        f"It is impossible to coerce {x} of type {type(x)} into a 1D vector. Please provide a scalar or a sequence of scalars."
    )
