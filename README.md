24-Hour Operations Optimization Suite

A framework for multi-objective optimization using genetic algorithms (NSGA-III and MOGA) to find efficient entry/exit points in 24-hour time-series data, balancing profitability and constraints.

## Core Features
- **Genetic Algorithm Strategies**: Implementation of NSGA-III and MOGA via the pymoo library.
- **Abstract Problem Definition**: A customizable OptimizationPoints class that handles binary selection across time-series data.
- **Constraint Management**: Specialized logic to ensure operational stability, including adjacency rules and sequence validation.
- **Batch Processing Pipeline**: Designed for analyzing historical data over long periods (days/months/years).

## Requirements
- Python 3.8+
- Libraries: numpy, pandas, matplotlib, pymoo

## Repository Structure
- `pipelines.py`: The entry point for batch analysis and historical backtesting.
- `strategies.py`: Contains the strategy design pattern implementations (NSGA-III, GA).
- `problems.py`: Defines the objective functions and the constraint environment.
- `tests/`: Directory containing unit tests.
- `plots/`: Directory where daily optimization results are saved.

## Getting Started

### 1. Installation
Ensure you have the required dependencies installed:

```bash
pip install numpy pandas matplotlib pymoo
```

### 2. Usage
To run a batch analysis over a specific timeframe:

```python
from pipelines import batch_analysis
from strategies import NSGA3Strategy
import pandas as pd

# read your data
df = pd.read_csv("data.csv")  # Columns: 'day', 'data_values'
strategy = NSGA3Strategy(pop_size=100)

# Execute backtest
annual_impact = batch_analysis(df, strategy, start_day=0, end_day=365, starting_metric=1000.0)
print(f"Total Annualized Metric: {annual_impact}")
```

### 3. Running Tests
To run the unit tests:

```bash
python -m pytest tests/
```