from typing import TYPE_CHECKING

from streamerate import stream  # type: ignore

from model import Model, State

if TYPE_CHECKING:
    from actions import Action
    from simulator import Simulator


class Agent:
    def __init__(self, x, y, simulator: "Simulator"):
        self.x: int = x
        self.y: int = y

        self.simulator: Simulator = simulator
        self.model: Model = self.simulator.model

        self.state: State = None  # type: ignore
        self.updated: bool = False
        self.next_state: State = None  # type: ignore

    def mark_as_not_updated(self):
        self.updated = False

    def calculate_next_state(self):
        if self.updated:
            return
        # default behaviour
        self.next_state = self.state
        actions = self.state.actions
        self.perform_actions(actions)

    def perform_actions(self, actions: list["Action"]):
        initial_next_state = self.next_state
        for action in actions:
            action.step(self)
            if self.next_state != initial_next_state:
                return

    def go_to_next_state(self):
        self.state = self.next_state

    def force_state(self, new_state: State):
        self.state = self.next_state = new_state
        self.updated = True
