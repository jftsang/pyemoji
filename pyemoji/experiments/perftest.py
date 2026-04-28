"""Version of the Ising model adapted for performance testing using timeit."""

from collections import deque

from .ising import IsingSim, model as ising_model


class IsingSimPerfTest(IsingSim):
    def post_step(self):  # override
        pass

    def produce_plots(self):  # override
        pass


def main() -> None:
    deque(maxlen=0).extend(IsingSimPerfTest(ising_model, tmax=100).run())


if __name__ == "__main__":
    main()
