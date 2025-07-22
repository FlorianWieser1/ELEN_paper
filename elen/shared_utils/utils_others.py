import os
import time
import shutil
import numpy as np
from typing import Optional, Callable
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

### TIMING ####################################################################
def runtime(prefix: str = "", start_time: Optional[float] = None) -> None:
    """
    Print the runtime of a process in a human-readable format.

    Parameters:
        prefix (str): A label to describe the runtime output.
        start_time (Optional[float]): The starting time (usually from time.time()). If None, the current time will be used.

    Returns:
        None
    """
    if start_time is None:
        start_time = time.time()  # If no start_time is provided, use the current time

    exe_time = time.time() - start_time
    days, remainder = divmod(exe_time, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"== Runtime '{prefix}': {int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s")

def func_timer(func: Callable) -> Callable:
    """
    A decorator to measure the execution time of a function.

    Parameters:
        func (Callable): The function whose runtime is to be measured.

    Returns:
        Callable: The wrapped function that will print its execution time after completion.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        runtime(f"{func.__name__} function", start_time)
        return result

    return wrapper

### WANDB #####################################################################
def calculate_regression_metrics(target, pred):
    """
    Calculates common regression metrics between target and predicted values.

    Args:
        target (array-like): The ground truth target values.
        pred (array-like): The predicted values.

    Returns:
        tuple: Containing Pearson correlation, Spearman correlation, R² score, MAE, and variance of predictions.
    """
    R, _ = pearsonr(target, pred)
    spearman, _ = spearmanr(target, pred)
    r2 = r2_score(target, pred)
    mae = mean_absolute_error(target, pred)
    var_out = np.var(pred)
    return R, spearman, r2, mae, var_out
