import abc
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from pyemoji.simulator import Simulator


class FileWriter(abc.ABC):
    def __init__(self, simulator: "Simulator", filename: str | Path):
        self.simulator: "Simulator" = simulator
        self.filename: Path = Path(filename)
        self.f: TextIO | None = None

    def __enter__(self):
        self.f = self.filename.open("a")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f is not None:
            self.f.close()

    @abc.abstractmethod
    def write_header(self):
        pass

    @abc.abstractmethod
    def write_state(self):
        pass


class PopulationFileWriter(FileWriter):
    def write_header(self):
        pop = self.simulator.populations()
        self.filename.write_text("time," + ",".join(pop.keys()) + "\n")

    def write_state(self):
        self.filename.write_text(
            f"{self.simulator.time},"
            + ",".join(map(str, self.simulator.populations().values()))
            + "\n"
        )
