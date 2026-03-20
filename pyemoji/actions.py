import operator
import random
from abc import ABC, abstractmethod
from functools import reduce
from typing import Annotated, Literal, Protocol, TYPE_CHECKING, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from simulator import Agent


class Action(ABC):
    @abstractmethod
    def step(self, agent: "Agent") -> None: ...


class IfNeighborAction(BaseModel, Action):
    type: Literal["if_neighbor"] = "if_neighbor"
    sign: Literal[">", ">=", "=", "<=", "<"]
    num: int
    stateID: int
    actions: list["AnyAction"] = Field(default_factory=list)

    def step(self, agent: "Agent"):
        neighbors = agent.simulator.get_neighbors(agent)
        desired_state = agent.rules.sid2state(self.stateID)
        count = len([x for x in neighbors if x.state == desired_state])
        cond: bool
        if self.sign == ">=":
            cond = count >= self.num
        else:
            raise NotImplementedError
        if cond:
            agent.perform_actions(self.actions)


class IfRandomAction(BaseModel, Action):
    type: Literal["if_random"] = "if_random"
    probability: float
    actions: list["AnyAction"] = Field(default_factory=list)

    def step(self, agent: "Agent"):
        x = random.uniform(0, 1)
        if x < self.probability:
            agent.perform_actions(self.actions)


class GoToStateAction(BaseModel, Action):
    type: Literal["go_to_state"] = "go_to_state"
    stateID: int

    def step(self, agent: "Agent"):
        agent.next_state = agent.rules.sid2state(self.stateID)


action_classes = [
    IfNeighborAction,
    IfRandomAction,
    GoToStateAction,
]

AnyAction = Annotated[
    reduce(operator.ior, action_classes),
    Field(discriminator="type"),
]


for ac in action_classes:
    ac.model_rebuild()
