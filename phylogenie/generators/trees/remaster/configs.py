import phylogenie.generators.configs as cfg
from phylogenie.generators.configs import StrictBaseModel


class ReactionConfig(StrictBaseModel):
    rate: cfg.SkylineParameterLikeConfig
    value: str


class PunctualReactionConfig(StrictBaseModel):
    times: cfg.ManyScalarsConfig
    value: str
    p: cfg.ManyScalarsConfig | None = None
    n: cfg.ManyIntsConfig | None = None
