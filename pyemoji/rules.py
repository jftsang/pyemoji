from typing import Any, Literal

import pydantic
from pydantic import Field

from actions import AnyAction


class Model(pydantic.BaseModel):
    @classmethod
    def from_dict(cls, *a, **k):
        return super().model_validate(*a, **k)


class WorldRules(Model):
    neighborhood: Literal["moore", "neumann"]
    proportions: list[dict[str, int]]  # TODO should be stateID and parts
    size: dict[str, int]  # TODO should be height and width
    update: Literal["simultaneous"] = "simultaneous"


class State(Model):
    id: int
    icon: str
    name: str
    actions: list[AnyAction] = Field(default_factory=list)

    def __hash__(self):
        return hash(self.id)


class Rules(Model):
    meta: dict[str, Any] = None  # metarules, TODO
    states: list[State]
    world: WorldRules

    @property
    def statemap(self) -> dict[int, State]:
        return {s.id: s for s in self.states}

    def sid2state(self, sid: int) -> State:
        return self.statemap[sid]

    def sid2char(self, sid: int) -> str:
        return self.statemap[sid].icon

    @property
    def default_state(self) -> State:
        return self.states[0]
