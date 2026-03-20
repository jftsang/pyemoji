import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from actions import GoToStateAction, IfNeighborAction, IfRandomAction
from rules import Rules, State, WorldRules
from simulator import Simulator

downstate = State(id=0, name="down", icon="⚫️", actions=[])
upstate = State(id=1, icon="🔴", name="up", actions=[])

decay = GoToStateAction(stateID=0)
mightdecay = IfRandomAction(probability=0.001, actions=[decay])

# Progressively more likely to decay if you have neighbours who have
# already decayed
act = upstate
for num in range(1, 9):
    prev = act
    act = IfNeighborAction(sign=">=", num=num, stateID=0, actions=[mightdecay])
    prev.actions.append(act)


upstate.actions.append(mightdecay)

rules = Rules(
    states=[downstate, upstate],
    world=WorldRules(
        neighborhood="moore",
        proportions=[{"stateID": 0, "parts": 0}, {"stateID": 1, "parts": 100}],
        size={"width": 31, "height": 29},
    ),
)

simulator = Simulator(rules)

pops = []

tmax = 2000
for t, s, p in tqdm(simulator.run(), total=tmax):
    pops.append({"t": t, **p})

    if p["up"] <= 10 or t > tmax:
        break

df = pd.DataFrame.from_records(pops)

ax = plt.gca()
ax.plot(df["t"], df["up"])
ax.plot(df["t"], 31 * 29 * np.exp(-0.001 * df["t"]), "--", label="model")
ax.set_yscale("log")
ax.legend()
plt.show()
