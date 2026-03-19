import json
from argparse import ArgumentParser
from pathlib import Path

from rules import Rules
from simulator import Simulator


def main(argv: str | None = None) -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "rules",
        type=Path,
        help="path to the rules configuration",
        default="models/ising.json",
    )
    args = parser.parse_args(argv)

    d = json.loads(args.rules.read_text())
    rules = Rules.from_dict(d)

    simulator = Simulator(rules)

    results = simulator.run()
    for t, s, p in results:
        if t > 10:
            break
        print(t, p)

    print(simulator)


if __name__ == "__main__":
    main()
