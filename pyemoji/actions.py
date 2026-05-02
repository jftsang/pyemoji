import operator
import random
from abc import ABC, abstractmethod
from functools import reduce
from typing import Annotated, Iterable, Literal, Sequence

import pydantic


class Action(ABC):
    @abstractmethod
    def step(self, agent: "Agent") -> None: ...


class IfNeighborAction(pydantic.BaseModel, Action):
    type: Literal["if_neighbor"] = "if_neighbor"
    sign: Literal[">", ">=", "=", "<=", "<"]
    num: int
    neighborState: "State"
    actions: list["AnyAction"] = pydantic.Field(default_factory=list)

    def step(self, agent: "Agent"):
        neighbors = agent.simulator.get_neighbors(agent)
        count = len([x for x in neighbors if x.state is self.neighborState])
        op = {
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
            "=": operator.eq,
        }[self.sign]
        cond: bool = op(count, self.num)
        if cond:
            agent.perform_actions(self.actions)


class IfRandomAction(pydantic.BaseModel, Action):
    type: Literal["if_random"] = "if_random"
    probability: float
    actions: list["AnyAction"] = pydantic.Field(default_factory=list)

    def step(self, agent: "Agent"):
        x = random.uniform(0, 1)
        if x < self.probability:
            agent.perform_actions(self.actions)


class GoToStateAction(pydantic.BaseModel, Action):
    type: Literal["go_to_state"] = "go_to_state"
    destState: "State"

    def step(self, agent: "Agent"):
        agent.next_state = self.destState


class MoveToAction(pydantic.BaseModel, Action):
    type: Literal["move_to"] = "move_to"
    dest: Literal["anywhere", "neighbors"]
    destState: "State"  # state that we can move into
    leaveState: "State"  # state that we leave behind
    # state that we change into (None means don't change)
    resultState: "State | None" = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def fix_types(cls, data: dict) -> dict:
        if "dest" not in data:
            s = int(data.pop("space"))  # raises KeyError
            data["dest"] = {0: "neighbors", 1: "anywhere"}[s]

        return data

    def step(self, agent: "Agent"):
        if self.dest == "anywhere":
            candidates: Iterable[Agent] = agent.simulator.get_all_agents()
        elif self.dest == "neighbors":
            candidates: Iterable[Agent] = agent.simulator.get_neighbors(agent)
        else:
            raise ValueError

        eligibles: Sequence[Agent] = [
            a for a in candidates if a.state is self.destState
        ]
        if not eligibles:
            return  # can't move anywhere, give up

        chosen: Agent = random.choice(eligibles)
        if self.resultState is None:
            chosen.force_state(agent.state)
        else:
            chosen.force_state(self.resultState)
        agent.next_state = self.leaveState


action_classes = [
    IfNeighborAction,
    IfRandomAction,
    GoToStateAction,
    MoveToAction,
]


union_type = reduce(operator.ior, action_classes)

AnyAction = Annotated[union_type, pydantic.Field(discriminator="type")]

# Imports happen down here to avoid circular imports but we need them to
# rebuild models

from pyemoji.agent import Agent  # noqa: E402
from pyemoji.model import State  # noqa: E402

for ac in action_classes:
    ac.model_rebuild()
