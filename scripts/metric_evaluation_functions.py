# scripts/metrics_functions.py

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy import signal
from scipy import stats

def pontius_equations(df, colx, coly):
    """
    Calculates a suite of comparison metrics for continuous data based on the work of Pontius.
    https://wordpress.clarku.edu/rpontius/wp-content/uploads/sites/884/2024/10/Pontius-Jr-2022-Metrics-That-Make-a-Difference.pdf
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        colx (str): The column name for the reference data (e.g., survey data).
        coly (str): The column name for the prediction data (e.g., model output).

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    y_mean = df[coly].mean()
    x_mean = df[colx].mean()

    # Mean Deviation (Systematic Error)
    md = y_mean - x_mean

    # Difference for each observation
    df['Di'] = df[coly] - df[colx]
    n = df.shape[0]

    # Mean Absolute Deviation
    mad = df['Di'].abs().sum() / n

    # Quantity Deviation
    qd = abs(df['Di'].sum()) / n

    # Allocation Deviation
    ad = (df['Di'].abs().sum() - abs(df['Di'].sum())) / n

    # Variance in X and Y
    var_x = ((df[colx] - x_mean)**2).sum() / n
    var_y = ((df[coly] - y_mean)**2).sum() / n # Note: Original code had x_mean here, corrected to y_mean

    # Correlation Coefficient (r) and R-squared (r2)
    r_numerator = ((df[colx] - x_mean) * (df[coly] - y_mean)).sum()
    r_denominator = np.sqrt((((df[colx] - x_mean)**2).sum()) * (((df[coly] - y_mean)**2).sum()))
    r = r_numerator / r_denominator
    r2 = r**2

    # Linear Regression (y = b*x + a)
    b = (((df[colx] - x_mean) * (df[coly] - y_mean)).sum()) / (((df[colx] - x_mean)**2).sum())
    a = y_mean - b * x_mean
    eq = f"y = {b:.2f}*x + {a:.2f}"

    # Root Mean Square Deviation (RMSD)
    rmsd = np.sqrt(((df[coly] - df[colx])**2).sum() / n)

    #  improved concordance coefficient of Willmott (dr)
    abs_diff_sum = (df['Di'].abs()).sum()
    abs_diff_from_mean_sum = (2 * (df[colx] - x_mean).abs().sum())
    
    if abs_diff_sum <= abs_diff_from_mean_sum:
        dr = 1 - (abs_diff_sum / abs_diff_from_mean_sum)
    else:
        dr = (abs_diff_from_mean_sum / abs_diff_sum) - 1

    return {
        "Reference Mean (x̄)": x_mean,
        "Prediction Mean (ȳ)": y_mean,
        "Mean Deviation (MD)": md,
        "Mean Absolute Deviation (MAD)": mad,
        "Quantity Deviation (QD)": qd,
        "Allocation Deviation (AD)": ad,
        "Variance in Reference (Var_x)": var_x,
        "Variance in Prediction (Var_y)": var_y,
        "Correlation (r)": r,
        "R-squared (r2)": r2,
        "Slope (b)": b,
        "Intercept (a)": a,
        "Linear Equation": eq,
        "Root Mean Square Deviation (RMSD)": rmsd,
        "Willmott (dr)": dr
    }



#Calculate some errors
def metrics_accuracy(df, colx, coly, decimals):
    """Calculates accuracy metrics for two columns in a DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        colx (str): The column name for the reference data (e.g., survey data).
        coly (str): The column name for the prediction data (e.g., model output).
        decimals (int): Number of decimal places to round the results.
    Returns:
        list: A list containing R-squared, RMSE, and MAE.
    """
    res = stats.linregress(df[colx], df[coly])
    r2 = round(res.rvalue**2, decimals)
    n = len(df[colx])
    p = 1  # Assuming only one predictor (colx)
    # adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)  # Adjusted R-squared formula. Not returning the radjust adjusted_r2
    rmse = round(np.sqrt(mean_squared_error(df[colx], df[coly])), decimals)
    mae = round(mean_absolute_error(df[colx], df[coly]), decimals)
    return [r2, rmse, mae]