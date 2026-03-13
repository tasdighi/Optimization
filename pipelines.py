import pandas as pd
import numpy as np
from strategies import OptimizationStrategy


def batch_analysis(data_values: pd.DataFrame, 
                   strategy: OptimizationStrategy, 
                   start_day: int, end_day: int, 
                   starting_metric: float) -> float:
    """
    Performs batch analysis on the input data using the specified optimization strategy.

    Args:
        data_values (pd.DataFrame): The input data containing values and timestamps.
        strategy (OptimizationStrategy): The optimization strategy to use.
        start_day (int): The starting index of the days to analyze.
        end_day (int): The ending index of the days to analyze.
        starting_metric (float): The initial metric value for optimization.

    Returns:
        float: The annual metric calculated from the optimization results.
    """
    if not isinstance(data_values, pd.DataFrame):
        raise ValueError("data_values must be a pandas DataFrame")
    if "day" not in data_values.columns or "data_values" not in data_values.columns:
        raise ValueError("DataFrame must contain 'day' and 'data_values' columns")
    unique_days = data_values["day"].unique()
    if start_day < 0 or end_day > len(unique_days) or start_day >= end_day:
        raise ValueError("Invalid start_day or end_day indices")
    
    annual_metric = 0.0
    for day in unique_days[start_day:end_day]:
        data = data_values[data_values["day"] == day]["data_values"].to_numpy()
        if len(data) == 0:
            continue  # Skip empty days
        result = strategy.optimize(data, starting_metric)
        state_of_metric = np.ones(len(data))  # Default state
        metric = strategy.problem.calculate_metric(starting_metric, state_of_metric)
        annual_metric += metric
        strategy.visualize(result, datetime=str(day), metric=metric)

    return annual_metric

