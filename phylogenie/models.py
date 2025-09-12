from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Distribution(BaseModel):
    type: str
    model_config = ConfigDict(extra="allow")

    @property
    def args(self) -> dict[str, Any]:
        assert self.model_extra is not None
        return self.model_extra
