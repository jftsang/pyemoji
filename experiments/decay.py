import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rules import Rules
from simulator import Simulator

d = json.loads((Path(__file__).parent / "decay.json").read_text())
rules = Rules.from_dict(d)

simulator = Simulator(rules)

pops = []

for t, s, p in tqdm(simulator.run(), total=2000):
    pops.append({"t": t, **p})

    if p["up"] <= 10 or t > 10000:
        break

df = pd.DataFrame.from_records(pops)

ax = plt.gca()
ax.plot(df["t"], df["up"])
ax.plot(df["t"], 1024 * np.exp(-0.005 * df["t"]), "--", label="model")
ax.set_yscale("log")
ax.legend()
plt.show()
