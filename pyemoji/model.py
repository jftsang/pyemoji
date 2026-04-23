from streamerate import stream
from typing import Any, Literal, Mapping

import pydantic

from pyemoji.actions import AnyAction


class WorldRules(pydantic.BaseModel):
    @pydantic.model_validator(mode="before")
    @classmethod
    def unpack_size(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise TypeError
        if "size" in data and "height" not in data and "width" not in data:
            s = data.pop("size")
            data = {**data, **s}

        if "proportions" in data:
            plist = data.get("proportions")
            if isinstance(plist, list):
                # [{"stateID": 0, "parts": 0}, {"stateID": 1, "parts": 100}]
                data["proportions"] = (
                    stream(plist).map(lambda d: (d["stateID"], d["parts"])).to_dict()
                )

        return data

    neighborhood: Literal["moore", "neumann"]
    proportions: Mapping[int, int | float]  # map of species IDs to proportions
    height: int
    width: int
    update: Literal["simultaneous"] = "simultaneous"


class State(pydantic.BaseModel):
    __slots__ = ["id", "icon", "name", "actions"]

    id: int
    icon: str
    name: str
    actions: list[AnyAction]

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return self.name


class Model(pydantic.BaseModel):
    meta: dict[str, Any] = None  # metarules, TODO  # ty:ignore[invalid-assignment]
    states: list[State]
    world: WorldRules

    @property
    def statemap(self) -> dict[int, State]:
        return {s.id: s for s in self.states}

    def sid2char(self, sid: int) -> str:
        return self.statemap[sid].icon

    @property
    def default_state(self) -> State:
        return self.states[0]
