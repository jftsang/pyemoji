import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from model import Model
from simulator import Simulator


def main(argv: str | None = None) -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "model",
        type=Path,
        help="path to the model configuration",
        default="models/ising.json",
    )
    args = parser.parse_args(argv)

    d = json.loads(args.rules.read_text())
    rules = Model.model_validate(d)

    simulator = Simulator(rules)

    pops = []

    for t, s, p in tqdm(simulator.run()):
        pops.append({"t": t, **p})

        # if t >= 1000:
        if p["up"] == 0:
            break

    df = pd.DataFrame.from_records(pops).to_csv("out.csv", index=False)


if __name__ == "__main__":
    main()
