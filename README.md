# Pyemoji

Python port of Nicky Case's Emoji Simulator

Original: https://ncase.me/sim/


## Installation

```bash
pip install -e ".[dev,gui]"
pytest
```

## Running

```bash
python -m pyemoji.experiments.ising
```

## New features

* Set up model rules in Python, not in JSON
* Compatible with loading JSON files from the original
* Arbitrary domain sizes
* Export experimental results as Pandas dataframes
* Customisable control over what happens between steps or at the end of the experiment (dependency inversion)

## To-do list

* Better serialization
* Ability to reload from aborted simulations
