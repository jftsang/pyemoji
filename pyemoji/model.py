from typing import Any, Literal

import pydantic

from pyemoji.actions import AnyAction


class WorldRules(pydantic.BaseModel):
    neighborhood: Literal["moore", "neumann"]
    proportions: list[dict[str, int]]  # TODO should be stateID and parts
    size: dict[str, int]  # TODO should be height and width
    update: Literal["simultaneous"] = "simultaneous"


class State(pydantic.BaseModel):
    id: int
    icon: str
    name: str
    actions: list[AnyAction] = pydantic.Field(default_factory=list)

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
