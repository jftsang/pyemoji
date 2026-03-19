import operator
from functools import reduce
from typing import Annotated, Literal, Protocol, TYPE_CHECKING, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from simulator import Agent


@runtime_checkable
class Action(Protocol):
    def step(self, agent: "Agent") -> None: ...


class IfNeighborAction(BaseModel):
    type: Literal["if_neighbor"]
    sign: Literal[">", ">=", "=", "<=", "<"]
    num: int
    stateID: int
    actions: list["AnyAction"] = Field(default_factory=list)

    def step(self, agent: "Agent"): ...


class IfRandomAction(BaseModel):
    type: Literal["if_random"]
    probability: float
    actions: list["AnyAction"] = Field(default_factory=list)


class GoToStateAction(BaseModel):
    type: Literal["go_to_state"]
    stateID: int

    def step(self, agent: "Agent"):
        agent.next_sid = self.stateID


action_classes = [IfNeighborAction, IfRandomAction, GoToStateAction]

AnyAction = Annotated[reduce(operator.ior, action_classes), Field(discriminator="type")]


for ac in action_classes:
    ac.model_rebuild()
