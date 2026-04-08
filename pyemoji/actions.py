import operator
import random
from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Annotated, Iterable, Literal

import pydantic

if TYPE_CHECKING:
    from pyemoji.agent import Agent


class Action(ABC):
    @abstractmethod
    def step(self, agent: "Agent") -> None: ...


class IfNeighborAction(pydantic.BaseModel, Action):
    type: Literal["if_neighbor"] = "if_neighbor"
    sign: Literal[">", ">=", "=", "<=", "<"]
    num: int
    stateID: int
    actions: list["AnyAction"] = pydantic.Field(default_factory=list)

    def step(self, agent: "Agent"):
        neighbors = agent.simulator.get_neighbors(agent)
        desired_state = agent.model.states[self.stateID]
        count = len([x for x in neighbors if x.state == desired_state])
        cond: bool
        if self.sign == ">=":
            cond = count >= self.num
        else:
            raise NotImplementedError
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
    stateID: int

    def step(self, agent: "Agent"):
        agent.next_state = agent.model.states[self.stateID]


class MoveToAction(pydantic.BaseModel, Action):
    type: Literal["move_to"] = "move_to"
    dest: Literal["anywhere", "neighbors"]
    spotStateID: int
    leaveStateID: int

    @pydantic.model_validator(mode="before")
    @classmethod
    def fix_types(cls, data: dict) -> dict:
        if isinstance(ssid := data["spotStateID"], str):
            data["spotStateID"] = int(ssid)
        if "dest" not in data:
            s = int(data.pop("space"))  # raises KeyError
            data["dest"] = {0: "neighbors", 1: "anywhere"}[s]

        return data

    def step(self, agent: "Agent"):
        candidates: Iterable["Agent"]
        if self.dest == "anywhere":
            candidates = agent.simulator.get_all_agents()
        elif self.dest == "neighbors":
            candidates = agent.simulator.get_neighbors(agent)
        else:
            raise ValueError

        eligibles: list[Agent] = [
            a for a in candidates if a.state.id == self.spotStateID
        ]
        if not eligibles:
            return  # can't move anywhere, give up

        chosen: "Agent" = random.choice(eligibles)
        chosen.force_state(agent.state)
        agent.next_state = agent.model.states[self.leaveStateID]


action_classes = [
    IfNeighborAction,
    IfRandomAction,
    GoToStateAction,
    MoveToAction,
]


union_type = reduce(operator.ior, action_classes)

AnyAction = Annotated[union_type, pydantic.Field(discriminator="type")]


for ac in action_classes:
    ac.model_rebuild()
