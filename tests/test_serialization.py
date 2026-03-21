from pyemoji.actions import GoToStateAction, IfRandomAction
from pyemoji.model import Model, State, WorldRules
from pyemoji.simulator import Simulator

downstate = State(id=0, name="down", icon="⚫️", actions=[])
upstate = State(id=1, icon="🔴", name="up", actions=[])

decay = GoToStateAction(stateID=0)

upstate.actions.append(decay)

rules = Model(
    states=[downstate, upstate],
    world=WorldRules(
        neighborhood="moore",
        proportions=[{"stateID": 0, "parts": 0}, {"stateID": 1, "parts": 100}],
        size={"width": 3, "height": 5},
    ),
)


simulator = Simulator(rules)


def test_dump_load():
    simulator.setup_ics()
    initial_dump = simulator.dump()
    assert initial_dump["time"] == 0
    assert initial_dump["grid"] == "111111111111111"
    simulator.step()

    d = simulator.dump()
    assert d["time"] == 1
    assert d["grid"] == "000000000000000"

    simulator.load(initial_dump)
    assert simulator.time == 0
    assert simulator.grid[0, 0].state == upstate
