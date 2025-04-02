import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_results(*exps):
    """
    Plots the accuracy/error comparison across multiple experiments for 
    absolute distance error, relative distance error, and relative direction accuracy.
    
    :param exps: List of experiment directories containing results.json
    """
    tasks = ["Absolute Distance Error", "Relative Distance Error", "Relative Direction Accuracy"]
    measures = [[] for _ in range(3)]  # Store measures for all tasks
    models = list(exps)  # Experiment names

    # Load results for all experiments
    for exp in exps:
        with open(f"predictions/{exp}/results.json", "r") as f:
            result = json.load(f)

        # Store results per task
        for task_idx, task_name in enumerate(tasks):
            task_result = result[list(result.keys())[task_idx]]
            measures[task_idx].append(task_result)  # Append to correct list

    # Iterate over tasks and plot
    for task_idx, task_name in enumerate(tasks):
        plt.figure(figsize=(10, 6))

        if isinstance(measures[task_idx][0], list):  # Grouped bar chart for relative direction
            measures_arr = np.array(measures[task_idx])  # Convert list of lists to NumPy array
            num_experiments, num_dimensions = measures_arr.shape
            bar_width = 0.2  # Width of bars
            x = np.arange(num_dimensions)  # X positions for categories

            for i in range(num_experiments):
                plt.bar(x + i * bar_width, measures_arr[i], width=bar_width, label=models[i])

            plt.xticks(x + bar_width * (num_experiments - 1) / 2, ["Left/Right", "Above/Below", "Front/Back"])
            plt.ylabel("Accuracy (%)")
            plt.title("Relative Direction Accuracy Comparison")

        else:  # Standard bar chart for absolute/relative distance error
            plt.bar(models, measures[task_idx])
            plt.ylabel("MAE" if task_idx < 2 else "Accuracy")
            plt.title(f"{task_name} Comparison")

        plt.xlabel("Experiments")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend() if isinstance(measures[task_idx][0], list) else None
        plt.show()

        print(f"Plotted: {task_name}")


def plot_abs_distance(*exps):
    all_labels = set()
    exp_errors = {}

    # Step 1: Collect all unique frame labels and store per-experiment errors
    for exp in exps:
        with open(f"predictions/{exp}/results.json", "r") as f:
            result = json.load(f)

        single_results = result["absolute_distance"]["single_results"]
        errors_dict = {}

        for biome, trajs in single_results.items():
            for traj, frames in trajs.items():
                for frame, error in frames.items():
                    label = f"{biome}_{traj}_{frame}"
                    all_labels.add(label)
                    errors_dict[label] = error

        exp_errors[exp] = errors_dict

    # Step 2: Sort all frame labels for consistent x-axis
    all_labels = sorted(all_labels)
    x = np.arange(len(all_labels))  # numeric x-axis positions
    bar_width = 0.8 / len(exps)  # shrink bar width to fit all experiments

    # Step 3: Plot grouped bars
    plt.figure(figsize=(14, 6))
    for i, exp in enumerate(exps):
        errors = exp_errors[exp]
        # Fill with 0.0 or np.nan for missing predictions
        y_values = [errors.get(label, 0.0) for label in all_labels]
        plt.bar(x + i * bar_width, y_values, width=bar_width, label=exp)

    # Step 4: Format plot
    plt.xticks(x + bar_width * (len(exps) - 1) / 2, all_labels, rotation=90, fontsize=8)
    plt.ylabel("Normalized Absolute Distance Error")
    plt.xlabel("Biome_Trajectory_Frame")
    plt.title("Normalized Absolute Distance Error Comparison Across Experiments")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()    


def plot_rel_distance(*exps):
    all_labels = set()
    exp_errors = {}

    # Step 1: Collect all unique frame labels and store per-experiment errors
    for exp in exps:
        with open(f"predictions/{exp}/results.json", "r") as f:
            result = json.load(f)

        single_results = result["relative_distance"]["single_results"]
        errors_dict = {}

        for biome, trajs in single_results.items():
            for traj, frames in trajs.items():
                for frame, error in frames.items():
                    label = f"{biome}_{traj}_{frame}"
                    all_labels.add(label)
                    errors_dict[label] = error

        exp_errors[exp] = errors_dict

    # Step 2: Sort all frame labels for consistent x-axis
    all_labels = sorted(all_labels)
    x = np.arange(len(all_labels))  # numeric x-axis positions
    bar_width = 0.8 / len(exps)  # shrink bar width to fit all experiments

    # Step 3: Plot grouped bars
    plt.figure(figsize=(14, 6))
    for i, exp in enumerate(exps):
        errors = exp_errors[exp]
        # Fill with 0.0 or np.nan for missing predictions
        y_values = [errors.get(label, 0.0) for label in all_labels]
        plt.bar(x + i * bar_width, y_values, width=bar_width, label=exp)

    # Step 4: Format plot
    plt.xticks(x + bar_width * (len(exps) - 1) / 2, all_labels, rotation=90, fontsize=8)
    plt.ylabel("Normalized Relative Distance Error")
    plt.xlabel("Biome_Trajectory_Frame")
    plt.title("Normalized Relative Distance Error Comparison Across Experiments")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_rel_direction(*exps):
    
    models = list(exps)
    measures = []
    
    for exp in exps:
        with open(f"predictions/{exp}/results.json", "r") as f:
            result = json.load(f)

        # Store results for the task
        task_result = result["relative_direction"]["relative_direction_accuracy"]
        measures.append(task_result)  # Append to correct list

    measures_arr = np.array(measures)

    num_experiments, num_dimensions = measures_arr.shape
    plt.figure(figsize=(10, 6))
    bar_width = 0.2  # Width of bars
    x = np.arange(num_dimensions)  # X positions for categories
    
    for i in range(num_experiments):
        plt.bar(x + i * bar_width, measures_arr[i], width=bar_width, label=models[i])

    plt.xticks(x + bar_width * (num_experiments - 1) / 2, ["Left/Right", "Above/Below", "Front/Back"])
    plt.ylabel("Accuracy (%)")
    plt.title("Relative Direction Accuracy Comparison")
    plt.xlabel("Experiments")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()


def plot_abs_distance_histogram(exp, num_bins=10):
    # Load results.json
    with open(f"predictions/{exp}/results.json", "r") as f:
        result = json.load(f)

    # Extract all absolute distance errors from single_results
    single_results = result["absolute_distance"]["single_results"]
    errors = []

    for biome, trajs in single_results.items():
        for traj, frames in trajs.items():
            for frame, error in frames.items():
                errors.append(error)

    if not errors:
        print(f"No errors found for experiment: {exp}")
        return

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=num_bins, color="skyblue", edgecolor="black")
    plt.xlabel("Absolute Distance Error")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution - {exp}")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_abs_distance_histogram_multi(*exp_names):
    """
    Plot normalized absolute distance error distributions for multiple experiments.

    Parameters:
    - exp_names: list of experiment folder names under 'predictions/'
    - bins: custom list of bin edges (e.g., [0, 0.5, 1, 1.5, ..., 5])
    """
    plt.figure(figsize=(10, 6))
    bins = [i * 0.5 for i in range(11)]
    for exp_name in exp_names:
        # Load results.json
        with open(f"predictions/{exp_name}/results.json", "r") as f:
            result = json.load(f)

        # Extract error list
        single_results = result["absolute_distance"]["single_results"]
        errors = []
        for biome in single_results:
            for traj in single_results[biome]:
                for frame in single_results[biome][traj]:
                    errors.append(single_results[biome][traj][frame])

        if not errors:
            print(f"No errors found in {exp_name}, skipping...")
            continue

        # Plot normalized histogram
        plt.hist(
            errors,
            bins=bins,
            density=False,
            alpha=0.6,
            label=exp_name,
            edgecolor="black",
            histtype="stepfilled"
        )

    plt.xlabel("Absolute Distance Error")
    plt.ylabel("Frequency")
    plt.title("Normalized Error Distribution Across Experiments")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rel_distance_histogram_multi(*exp_names):
    """
    Plot normalized relative distance error distributions for multiple experiments.

    Parameters:
    - exp_names: list of experiment folder names under 'predictions/'
    - bins: custom list of bin edges (e.g., [0, 0.5, 1, 1.5, ..., 5])
    """
    plt.figure(figsize=(10, 6))
    bins = [i * 0.5 for i in range(11)]
    for exp_name in exp_names:
        # Load results.json
        with open(f"predictions/{exp_name}/results.json", "r") as f:
            result = json.load(f)

        # Extract error list
        single_results = result["relative_distance"]["single_results"]
        errors = []
        for biome in single_results:
            for traj in single_results[biome]:
                for frame in single_results[biome][traj]:
                    errors.append(single_results[biome][traj][frame])

        if not errors:
            print(f"No errors found in {exp_name}, skipping...")
            continue

        # Plot normalized histogram
        plt.hist(
            errors,
            bins=bins,
            density=False,
            alpha=0.6,
            label=exp_name,
            edgecolor="black",
            histtype="stepfilled"
        )

    plt.xlabel("Relative Distance Error")
    plt.ylabel("Frequency")
    plt.title("Normalized Error Distribution Across Experiments")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Old Plotter
    # plot_results("20250303_231817_gpt", "20250303_231817_gpt_socratic")
    
    
    plot_abs_distance("20250303_231817_gpt_v8", "20250303_231817_gpt_socratic_v2", "20250303_231817_llava")
    plot_rel_distance("20250303_231817_gpt_v8", "20250303_231817_gpt_socratic_v2", "20250303_231817_llava")
    plot_rel_direction("20250303_231817_gpt_v8", "20250303_231817_gpt_socratic_v2", "20250303_231817_llava")



    #plot_abs_distance_histogram("20250303_231817_gpt_v8")

    plot_abs_distance_histogram_multi("20250303_231817_gpt_v8", "20250303_231817_gpt_socratic_v2", "20250303_231817_llava")
    plot_rel_distance_histogram_multi("20250303_231817_gpt_v8", "20250303_231817_gpt_socratic_v2", "20250303_231817_llava")