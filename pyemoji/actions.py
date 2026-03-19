import operator
from functools import reduce
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class ActionProtocol(Protocol):
    def step(self) -> None: ...


class IfNeighborAction(BaseModel):
    type: Literal["if_neighbor"]
    sign: Literal[">", ">=", "=", "<=", "<"]
    num: int
    stateID: int
    actions: list["AnyAction"] = Field(default_factory=list)


class IfRandomAction(BaseModel):
    type: Literal["if_random"]
    probability: float
    actions: list["AnyAction"] = Field(default_factory=list)


class GoToStateAction(BaseModel):
    type: Literal["go_to_state"]
    stateID: int


action_classes = [IfNeighborAction, IfRandomAction, GoToStateAction]

AnyAction = Annotated[reduce(operator.ior, action_classes), Field(discriminator="type")]


for ac in action_classes:
    ac.model_rebuild()
