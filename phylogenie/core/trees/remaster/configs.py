import phylogenie.core.configs as cfg
from phylogenie.configs import StrictBaseModel


class ReactionConfig(StrictBaseModel):
    rate: cfg.SkylineParameterLikeConfig
    value: str


class PunctualReactionConfig(StrictBaseModel):
    times: cfg.ManyScalarsConfig
    value: str
    p: cfg.ManyScalarsConfig | None = None
    n: cfg.ManyIntsConfig | None = None
