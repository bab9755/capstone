from multiprocessing import Pool

import polars as pl

from vi import Agent, Config, HeadlessSimulation, Matrix, Simulation


def run_simulation(config: Config) -> pl.DataFrame:
    return (
        HeadlessSimulation(config)
        .batch_spawn_agents(100, Agent, ["images/white.png"])
        .run()
        .snapshots
    )


if __name__ == "__main__":
    # We create a threadpool to run our simulations in parallel
    with Pool() as p:
        print("Running simulations in parallel")
        # The matrix will create four unique configs
        matrix = Matrix(radius=[25, 50], seed=[1, 2])

        # Create unique combinations of matrix values
        configs = matrix.to_configs(Config)

        # Combine our individual DataFrames into one big DataFrame
        df = pl.concat(p.map(run_simulation, configs))
        print("Simulations completed")
        print(df)