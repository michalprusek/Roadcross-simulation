import sys
import argparse
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
import os
import logging
from datetime import datetime
from tqdm import tqdm  # Importing tqdm for progress bar
import importlib.util

# Nastavení dummy zvukového ovladače pro pygame na serveru bez zvukové karty
os.environ["SDL_AUDIODRIVER"] = "dummy"

# -----------------
# Nastavení logging
# -----------------
log_filename = f"grid_search_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

logging.info("Starting grid search for IDM parameters...")

# -----------------
# METRIC COMPUTATION
# -----------------
def compute_histogram_distance(simulated_data, real_data, num_bins=30):
    simulated_data = np.array(simulated_data)
    real_data = np.array(real_data)

    # Check for crash condition
    if len(simulated_data) < MAX_CARS - 1:
        logging.warning("Crash detected: simulated_data is shorter than real_data minus one.")
        return 100  # Large penalty value

    combined = np.concatenate((simulated_data, real_data))
    bin_edges = np.linspace(min(combined), max(combined) + 1e-9, num_bins + 1)
    sim_hist, _ = np.histogram(simulated_data, bins=bin_edges, density=True)
    real_hist, _ = np.histogram(real_data, bins=bin_edges, density=True)
    eps = 1e-9
    chi_square = np.sum(((sim_hist - real_hist) ** 2) / (real_hist + eps))
    return chi_square

# -----------------
# SIMULATION WRAPPER
# -----------------
def run_simulation_and_get_metric(
        real_data, min_gap, max_acc, max_dec, react_time, desired_speed, delta, n_simulations=5
):
    from straight_line import Simulation, Config
    from copy import deepcopy

    metrics = []
    for _ in range(n_simulations):
        config_for_this_run = deepcopy(Config())
        config_for_this_run.MIN_GAP = min_gap
        config_for_this_run.MAX_ACCELERATION = max_acc
        config_for_this_run.MAX_DECELERATION = max_dec
        config_for_this_run.REACT_TIME = react_time
        config_for_this_run.DESIRED_SPEED = desired_speed
        config_for_this_run.DELTA = delta
        config_for_this_run.MAX_CARS = MAX_CARS

        sim = Simulation(silent=True, config=config_for_this_run)
        sim.run()

        simulated_roadcross_intervals = sim.data_manager.roadcross_time_intervals
        metric_value = compute_histogram_distance(simulated_roadcross_intervals, real_data)
        metrics.append(metric_value)

    average_metric = np.mean(metrics)  # Calculate the average metric
    return average_metric

# -----------------
# HELPER FUNCTION FOR MULTIPROCESSING
# -----------------
def evaluate_combination(args):
    """Helper function to evaluate one combination in parallel."""
    real_data, params = args
    metric_val = run_simulation_and_get_metric(real_data=real_data, **params)

    logging.info(
        f"Combination: min_gap={params['min_gap']:.3f}, max_acc={params['max_acc']:.3f}, max_dec={params['max_dec']:.3f}, "
        f"react_time={params['react_time']:.3f}, desired_speed={params['desired_speed']:.3f}, delta={params['delta']}, "
        f"average_metric={metric_val:.4f}"
    )

    return {"params": params, "metric": metric_val}

# -----------------
# GRID SEARCH OPTIMIZER
# -----------------
def optimize_idm_parameters_grid(real_data, param_grid, n_jobs=None):
    param_names = sorted(param_grid.keys())
    param_lists_in_order = [param_grid[param] for param in param_names]
    all_combinations = list(product(*param_lists_in_order))
    param_dicts = [
        {name: value for name, value in zip(param_names, combo)}
        for combo in all_combinations
    ]

    n_jobs = n_jobs or cpu_count()
    results = []
    with Pool(processes=n_jobs) as pool:
        for result in tqdm(pool.imap(evaluate_combination, [(real_data, params) for params in param_dicts]), total=len(param_dicts)):
            results.append(result)

    best_result = min(results, key=lambda x: x["metric"])

    logging.info("\n--- BEST RESULT FOUND (Grid Search) ---")
    logging.info(f"Best metric value: {best_result['metric']:.4f}")
    logging.info("Best parameters:")
    for k, v in best_result["params"].items():
        logging.info(f"  {k}: {v}")

    return best_result

# -----------------
# MAIN EXECUTION
# -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Grid Search for IDM Parameters")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of CPU cores to use for parallel grid search (default: 1).")
    parser.add_argument("--excel_path", type=str, default="/mnt/lustre/helios-home/prusemic/AISee/SSI/real_data.xlsx",
                        help="Path to the Excel file containing real data in the first column.")
    args = parser.parse_args()

    n_jobs = args.n_jobs
    file_path = args.excel_path

    df = pd.read_excel(file_path, header=None)
    real_data = df.iloc[:, 0].values

    num_points = 5
    min_gap_values = np.linspace(0.5, 5, num_points).tolist()
    max_acc_values = np.linspace(1.0, 4.0, 4).tolist()
    max_dec_values = np.linspace(1.0, 5.0, num_points).tolist()
    react_time_values = np.linspace(0.3, 2.0, num_points).tolist()
    desired_speed_values = [8.33, 13.88, 19.44, 25]
    delta_values = [2, 3, 4, 5]
    MAX_CARS = 1000

    param_grid = {
        "min_gap": min_gap_values,
        "max_acc": max_acc_values,
        "max_dec": max_dec_values,
        "react_time": react_time_values,
        "desired_speed": desired_speed_values,
        "delta": delta_values
    }
    logging.info("Run optimization...")
    result = optimize_idm_parameters_grid(
        real_data=real_data,
        param_grid=param_grid,
        n_jobs=n_jobs
    )

    logging.info("\nGrid Search finished.")
    logging.info(f"Best metric: {result['metric']}")
    logging.info(f"Best parameters: {result['params']}")
