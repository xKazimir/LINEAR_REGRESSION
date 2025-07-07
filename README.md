# Linear Regression from Scratch

This repository contains a Python implementation of simple linear regression using gradient descent and analytical methods, without relying on high-level machine learning libraries. It includes:

- A dataset reader and visualizer (`show_data`) using Seaborn/Matplotlib.
- Custom implementations of the sum of squared errors (`SSR`), its gradients (`SSR_der`), and variance (`var`).
- A gradient descent optimizer (`fit_line_to_data`) to fit the line \(y = ax + b\).
- Calculation of performance metrics: \(R^2\) and F-statistic.
- Comparison against NumPy's built-in `np.polyfit`.

---

## Dataset

- **`linear_regression_dataset.csv`** and **`linear_regression_dataset_2.csv`** (and optionally `linear_regression_dataset_3.csv`): CSV files with two columns:
  - `x`: input feature
  - `y`: target value

- feel free to generate your own dataset and try it out

## Requirements
- Python 3.7+
- pandas
- numpy
- seaborn
- matplotlib

```bash
pip3 install pandas numpy seaborn matplotlib
```

## Usage

Clone the repository:
```bash
git clone https://github.com/yourusername/linear-regression-scratch.git
cd linear-regression-scratch
```
Run the main script:
```bash
python main.py --data linear_regression_dataset_2.csv
```

This will:

Display the scatter plot with the intercept-only line.
Fit a line via gradient descent, showing updates every 100 iterations.
Compare to NumPy's polyfit.
Print metrics: variances, 
R^2, F-statistic (and p-value).

Modify hyperparameters in main.py:
learning_rate (default 0.0003)
Maximum iterations (default 10,000)
Convergence threshold (eps)

## Code Structure
- main.py: orchestrates data loading, visualization, model fitting, and metrics.
- show_data(data, a, b): plots scatter and regression line.
- SSR(data, a, b): computes sum of squared residuals.
- SSR_der(data, a, b, is_for_intercept): gradient of SSR w.r.t. intercept (b) or slope (a).
- fit_line_to_data(data): gradient descent loop returning (a, b).
- var(data, a, b): mean squared error (variance of residuals).
