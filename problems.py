import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List
from pymoo.core.problem import ElementwiseProblem


class OptimizationPoints(ElementwiseProblem):
    """
    A generic binary-selection multi-objective problem
    This class defines the optimization problem for identifying low and high points
    in a given dataset based on specific constraints.
    
    Attributes:
        data_values (np.ndarray): The input data values for optimization.
        starting_metric (float): The initial metric value for the optimization.
    """
    def __init__(self, data_values: np.ndarray, starting_metric: float):
        """
        Initializes the OptimizationPoints class with input data and starting metric.

        Args:
            data_values (np.ndarray): The input data values for optimization.
            starting_metric (float): The initial metric value for the optimization.
        """
        super().__init__(n_var=len(data_values) * 2,
                         n_obj=2,
                         n_ieq_constr=0,
                         n_eq_constr=5,
                         xl=0,
                         xu=1
                         )
        self.data_values = data_values - np.max(data_values)
        self.minus_data_values = -1 * data_values
        self.starting_metric = starting_metric
        self.result = None

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates the optimization problem by calculating objectives and constraints.

        Args:
            x: The decision variables.
            out: The output dictionary to store objectives and constraints.
        """
        out["F"] = self.calculate_objectives(x)
        out["H"] = self.calculate_constraints(x)

    def calculate_objectives(self, x):
        """
        Calculates the objective functions.

        Args:
            x: The decision variables.

        Returns:
            np.ndarray: The calculated objective values.
        """
        f1 = np.sum(self.data_values * x[:self.n_var // 2] * self.low_points())
        f2 = np.sum(self.minus_data_values * x[self.n_var // 2:] * self.high_points())
        return np.column_stack([f1, f2])

    def calculate_constraints(self, x):
        """
        Calculates the constraints.

        Args:
            x: The decision variables.

        Returns:
            np.ndarray: The calculated constraint values.
        """
        nonzero_index_low = np.nonzero(x[:self.n_var // 2])
        nonzero_index_high = np.nonzero(x[self.n_var // 2:])

        h1 = 2 - len(nonzero_index_low[0])
        h2 = 2 - len(nonzero_index_high[0])

        h1_1 = 1 if h1 == 0 and 1 in np.diff(nonzero_index_low) else 0
        h2_1 = 1 if h2 == 0 and 1 in np.diff(nonzero_index_high) else 0

        h3 = 1
        if (h1 == h2 == 0
                and nonzero_index_low[0][0] < nonzero_index_high[0][0] - 2
                and nonzero_index_high[0][0] + 2 < nonzero_index_low[0][1] < nonzero_index_high[0][1] - 2):
            h3 = 0

        return np.column_stack([h1, h1_1, h2, h2_1, h3])

    def low_points(self):
        """
        Identifies low points in the dataset based on relative values.

        Returns:
            np.ndarray: An array indicating the low points in the dataset.
        """
        n = len(self.data_values)
        relative_value = np.zeros(n)

        for i in range(1, n - 1):
            if self.data_values[i] < self.data_values[i - 1] and self.data_values[i] <= self.data_values[i + 1]:
                relative_value[i] = 1

        return relative_value

    def high_points(self):
        """
        Identifies high points in the dataset based on relative values.

        Returns:
            np.ndarray: An array indicating the high points in the dataset.
        """
        n = len(self.data_values)
        data_values = -1 * self.data_values
        relative_value = np.zeros(n)

        for i in range(1, n - 1):
            if data_values[i] < data_values[i - 1] and data_values[i] <= data_values[i + 1]:
                relative_value[i] = 1

        return relative_value

    def calculate_metric(self, capacity: float, state_of_metric: np.ndarray) -> float:
        """
        Calculates the metric based on the optimization results.

        Args:
            capacity (float): The capacity value for scaling the metric.
            state_of_metric (np.ndarray): The state of the metric for each data point.

        Returns:
            float: The calculated metric value.
        """
        if state_of_metric is None or len(state_of_metric) == 0:
            state_of_metric = np.ones(len(self.data_values))
        return (np.sum(self.result.X[0, len(self.data_values):] * self.data_values * state_of_metric)
                - np.sum(self.result.X[0, :len(self.data_values)] * self.data_values * state_of_metric)) * capacity

    def visualize(self, plot: bool = False, datetime: str = None, metric: float = None) -> None:
        """
        Visualizes the optimization results, including low and high points.

        Args:
            plot (bool): Whether to display the plot.
            datetime (str): The datetime string for labeling the plot.
            metric (float): The metric value to display in the title.

        """
        low_index = np.nonzero(self.result.X[0][:self.n_var // 2])
        high_index = np.nonzero(self.result.X[0][self.n_var // 2:])
        low_values = [self.data_values[i] for i in low_index]
        high_values = [self.data_values[i] for i in high_index]

        plt.figure()
        plt.plot(self.data_values, label="data_values")
        plt.plot(low_index, low_values, "o", label="low points")
        plt.plot(high_index, high_values, "*", label="high points")

        plt.title("Metric on " + datetime + " is: " + str(metric))
        plt.xlabel("Time")
        plt.ylabel("Data Values")
        plt.legend()

        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/' + datetime + '.png')
        if plot:
            plt.show()

    def optimize(self, strategy):
        """
        Evaluates the optimization problem using the specified strategy.

        Args:
            strategy: The optimization strategy to use.
        """
        return strategy.optimize(self.data_values)
