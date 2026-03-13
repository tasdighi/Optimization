from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moga import MOGA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.util.ref_dirs import get_reference_directions
from problems import OptimizationPoints
import pandas as pd
import numpy as np

class OptimizationStrategy:
    """
    Base class for optimization strategies.
    """
    def __init__(self):
        self.problem = None

    def optimize(self, data_values: np.ndarray, 
                 starting_metric: float):
        """
        Abstract method to perform optimization.

        Args:
            data_values (np.ndarray): The input data for optimization.
            starting_metric (float): The initial metric value for optimization.

        Returns:
            The result of the optimization process.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def visualize(self, result, datetime: str, metric: float):
        """
        Visualizes the optimization results.

        Args:
            result: The optimization result.
            datetime (str): The datetime string for labeling.
            metric (float): The metric value.
        """
        if self.problem:
            self.problem.visualize(datetime=datetime, metric=metric)

class NSGA3Strategy(OptimizationStrategy):
    """
    Concrete implementation of the NSGA-III optimization strategy.
    """
    def __init__(self, pop_size: int):
        super().__init__()
        self.ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        self.algorithm = NSGA3(
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True,
            ref_dirs=self.ref_dirs
        )

    def optimize(self, data_values: np.ndarray, 
                 starting_metric: float):
        """
        Perform optimization using NSGA-III.

        Args:
            data_values (np.ndarray): The input data for optimization.
            starting_metric (float): The initial metric value for optimization.

        Returns:
            The result of the optimization process.
        """
        self.problem = OptimizationPoints(data_values, starting_metric)
        result = minimize(self.problem, self.algorithm)
        self.problem.result = result
        return result

class GAStrategy(OptimizationStrategy):
    """
    Implements the Multi-Objective Genetic Algorithm (MOGA) for optimization. This strategy
    is suitable for multi-objective optimization problems.
    """
    def __init__(self, pop_size: int):
        super().__init__()
        self.algorithm = MOGA(
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )
        
    def optimize(self, data_values: np.ndarray,
                starting_metric: float):
        """
        Run the Multi-Objective Genetic Algorithm (MOGA) on the given data.

        Args:
            data_values (np.ndarray): The input data for optimization.
            starting_metric (float): The initial metric value for optimization.

        Returns:
            The result of the optimization process, including decision variables and objectives.
        """
        self.problem = OptimizationPoints(data_values, starting_metric)
        result = minimize(self.problem, self.algorithm)
        self.problem.result = result
        return result