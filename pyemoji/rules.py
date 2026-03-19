from abc import ABC
from typing import Any, ClassVar, Literal

import pydantic
from pydantic import Field


class Model(pydantic.BaseModel):
    @classmethod
    def from_dict(cls, *a, **k):
        return super().model_validate(*a, **k)


class WorldRules(Model):
    neighborhood: Literal["moore", "neumann"]
    proportions: list[dict[str, int]]  # TODO should be stateID and parts
    size: dict[str, int]  # TODO should be height and width
    update: Literal["simultaneous"] = "simultaneous"


class Action(Model, ABC):
    acttype: ClassVar[str] = Field(alias="type")


class IfNeighborAction(Action):
    acttype = "if_neighbor"
    sign: Literal[">", ">=", "==", "<=", "<", "!="]
    num: int
    stateID: int
    actions: list[Action] = Field(default_factory=list)


class IfRandomAction(Action):
    acttype = "if_random"
    probability: float
    actions: list[Action]


class State(Model):
    id: int
    icon: str
    name: str
    actions: list[Action] = Field(default_factory=list)


class Rules(Model):
    meta: dict[str, Any] = None  # metarules, TODO
    states: list[State]
    world: WorldRules

    @property
    def statemap(self) -> dict[int, State]:
        return {s.id: s for s in self.states}

    def sid2char(self, sid: int) -> str:
        return self.statemap[sid].icon
