import phylogenie.core.configs as cfg
from phylogenie.configs import StrictBaseModel


class ReactionConfig(StrictBaseModel):
    rate: cfg.SkylineParameterLikeConfig
    value: str


class PunctualReactionConfig(StrictBaseModel):
    times: cfg.OneOrManyScalarsConfig
    value: str
    p: cfg.OneOrManyScalarsConfig | None = None
    n: cfg.OneOrManyIntsConfig | None = None
