"""
This file verifies the dimension findings by comparing three model-pairs
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)



model_pairs = [
    (
        "Hermes 3 70B",
        "NousResearch__Hermes-3-Llama-3.1-70B",
        "mlabonne__Hermes-3-Llama-3.1-70B-lorablated",
    ),
    (
        "Llama 3.1 8B Instruct",
        "meta-llama__Llama-3.1-8B-Instruct",
        "mlabonne__Meta-Llama-3.1-8B-Instruct-abliterated",
    ),
    (
        "Phi 3 Medium",
        "microsoft__Phi-3-medium-4k-instruct",
        "cognitivecomputations__dolphin-2.9.2-Phi-3-Medium-abliterated",
    )
]

discrimination_cols = ["Discrimination 0", "Discrimination 1"]
line_colors =  ["#0072B2", "#D55E00", "#009E73"]

def cohens_d(a, b):
    """
    Inputs:
        a | np.array: The first set of data
        b | np.array: The second set of data
    Output:
        Cohen's d
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan

    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)

    if pooled_var == 0:
        return np.nan

    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))

def compare_pair_on_set(responses, item_ids, pair_name, model_a, model_b, set_name):
    """
    Run an unpaired t-test and Cohen's d for one model pair on one item set

    Inputs:
        responses | pd.DataFrame: Per-item model scores
        item_ids | pd.Index: The question_ids in this item set
        pair_name | str: The display name of the model pair
        model_a | str: The base model name
        model_b | str: The uncensored variant model name
        set_name | str: The display name of the item set

    Outputs:
        A dict with one row of results
    """
    col_a = "acc_norm_" + model_a
    col_b = "acc_norm_" + model_b

    for col in (col_a, col_b):
        if col not in responses.columns:
            raise KeyError(f"Column {col!r} is not in the responses file")

    scores = responses.loc[responses.index.isin(item_ids), [col_a, col_b]].dropna()
    a = scores[col_a].to_numpy(dtype=float)
    b = scores[col_b].to_numpy(dtype=float)

    row = {
        "model_pair": pair_name,
        "item_set": set_name,
        "n_items_in_set": len(item_ids),
        "n_items_scored": len(scores),
        "model_a": model_a,
        "model_b": model_b,
        "mean_a": a.mean() if len(a) else np.nan,
        "mean_b": b.mean() if len(b) else np.nan,
        "std_a": a.std(ddof=1) if len(a) > 1 else np.nan,
        "std_b": b.std(ddof=1) if len(b) > 1 else np.nan,
        "mean_difference": a.mean() - b.mean() if len(a) else np.nan,
    }

    t_statistic, p_value = ttest_ind(a, b, equal_var=True)
    row["t_statistic"] = float(t_statistic)
    row["p_value"] = float(p_value)
    row["degrees_of_freedom"] = len(a) + len(b) - 2

    row["cohens_d"] = cohens_d(a, b)

    return row

def dimension_curves(item_stats, responses, column, n_bins):
    """
    Mean score for every model in every pair, within equal-count bins of one
    discrimination dimension
    Inputs:
        item_stats | pd.DataFrame: Item statistics indexed by question_id
        responses | pd.DataFrame: Per-item model scores 
        column | str: The discrimination dimension to bin, e.g. "Discrimination 0"
        n_bins | int: The number of equal-count bins to group items into

    Outputs:
        A pd.DataFrame with one row per bin per model, carrying the bin, the mean
        discrimination in that bin, the item count, the mean score and its
        standard error
    """
    shared = item_stats.index.intersection(responses.index)
    values = item_stats.loc[shared, column]
    scores = responses.loc[shared]

    bins = pd.qcut(values, n_bins, labels=False, duplicates="drop")
    bin_value = values.groupby(bins).mean()
    bin_items = values.groupby(bins).size()

    rows = []
    for pair_index, (pair_name, model_a, model_b) in enumerate(model_pairs):
        for role, model in (("base", model_a), ("variant", model_b)):
            column_name = "acc_norm_" + model

            mean = scores[column_name].groupby(bins).mean()

            rows.append(pd.DataFrame({
                "dimension": column,
                "bin": mean.index,
                "discrimination": bin_value.to_numpy(),
                "n_items": bin_items.to_numpy(),
                "model_pair": pair_name,
                "pair_index": pair_index,
                "role": role,
                "model": model,
                "mean_score": mean.to_numpy(),
            }))

    return pd.concat(rows, ignore_index=True)

def graph_dimension(curves, column, output_dir):
    """
    Graph every model pair on one discrimination dimension

    Inputs:
        curves | pd.DataFrame: The binned curves, as built by dimension_curves()
        column | str: The discrimination dimension graphed, e.g. "Discrimination 0"
        output_dir | str: The location to save the graph, ending with /

    Outputs:
        Saves the graph to {output_dir}performance_vs_{dimension}_model_pairs.png
    """
    dimension_curve = curves[curves["dimension"] == column]

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))

    n_bins = dimension_curve["bin"].nunique()
    dense = True

    for pair_index, (pair_name, _, _) in enumerate(model_pairs):
        color = line_colors[pair_index % len(line_colors)]
        for role, style, marker in (("base", "-", "o"), ("variant", "--", "D")):
            line = dimension_curve[
                (dimension_curve["pair_index"] == pair_index)
                & (dimension_curve["role"] == role)
            ].sort_values("bin")

            label = pair_name if role == "base" else f"{pair_name} abliterated"
            ax.plot(
                line["discrimination"], line["mean_score"],
                linestyle=style, marker=None if dense else marker, markersize=6,
                color=color, lw=1.4 if dense else 2, label=label
            )

    ax.set_xlabel(f"Dimension {column[-1]} Discrimination, ({n_bins} quantile bins)")
    ax.set_ylabel("Model performance (mean score)")
    # ax.set_title(f"Performance vs {column}")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    stem = column.lower().replace(" ", "_")
    plt.savefig(
        f"{output_dir}performance_vs_{stem}_model_pairs.png",
        dpi=400, bbox_inches="tight"
    )
    plt.close(fig)


def main(item_statistics, all_responses, output_dir, percent = 10, n_bins=100):
    """
    Inputs:
        all_responses | str: The original responses file
        item_statistics | str: The file with the item level results 
        output_dir | str: The folder in which to save the results
        percent | int: The percent tail of the dimensions to examine
        n_bins | int: The number of bins to graph
    Outputs:
        A csv with t-test and cohens d results, a graph for each dimension
    """
    responses = pd.read_csv(all_responses, index_col="question_id")
    item_stats = pd.read_csv(item_statistics, index_col="question_id")

    results = []
    memberships = []
    for column in discrimination_cols:
        values = item_stats[column]
        min_val, max_val = float(values.quantile(percent / 100)), float(values.quantile(1 - percent / 100))

        for side in ["low", "high"]:
            threshold = min_val if side == "low" else max_val
            set_name = f"{column} {side} {percent}%"

            if side == "low":
                mask = values <= min_val
            elif side == "high":
                mask = values >= max_val
            item_ids = item_stats.index[mask]

            for question_id in item_ids:
                memberships.append({"question_id": question_id, "item_set": set_name})

            for pair_name, model_a, model_b in model_pairs:
                row = compare_pair_on_set(
                    responses, item_ids, pair_name, model_a, model_b, set_name
                )
                row["dimension"] = column
                row["side"] = side
                row["cut_point"] = threshold
                row["percent"] = percent
                results.append(row)
                print(
                    f"  {pair_name}: "
                    f"n={row['n_items_scored']}, "
                    f"base={row['mean_a']:.4f}, variant={row['mean_b']:.4f}, "
                    f"t={row['t_statistic']:.4f}, p={row['p_value']:.4g}, "
                    f"d={row['cohens_d']:.4f}"
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}discrimination_pair_tests.csv", index=False)

    memberships_df = pd.DataFrame(memberships)
    memberships_df.to_csv(f"{output_dir}discrimination_item_sets.csv", index=False)

    # One graph per dimension, with all three model pairs on it
    curves = pd.concat(
        [dimension_curves(item_stats, responses, column, n_bins)
         for column in discrimination_cols],
        ignore_index=True
    )
    curves.to_csv(f"{output_dir}discrimination_pair_curves.csv", index=False)

    for column in discrimination_cols:
        graph_dimension(curves, column, output_dir)
        stem = column.lower().replace(" ", "_")
        print(f"\nSaved {output_dir}performance_vs_{stem}_model_pairs.png")

main("../final_item_statistics_0.csv", "../prompts_filtered.csv", "./runs/")