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
    print(rules)

    simulator = Simulator(rules)
    ...


if __name__ == "__main__":
    main()
