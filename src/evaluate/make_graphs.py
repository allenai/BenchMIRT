"""
This file creates the graphs of other analyses
"""
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import math

all_benchmarks = {
    "bbh": "subcategory_one",
    "bbq": "subcategory_one",
    "do_anything_now": "subcategory_one",
    "gpqa": "subcategory_one",
    "harmbench": "subcategory_one",
    "ifeval": "subcategory_one",
    "math": "subcategory_one",
    "mmlu-pro": "subcategory_one",
    "musr": "subcategory_one",
    "strongreject": "subcategory_one",
    "toxigen": "subcategory_one",
    "trustllm_jailbreaktrigger": "subcategory_one",
    "wildguardtest": "subcategory_two",
    "wildjailbreak": "subcategory_one",
    "wmdp": "subcategory_one",
    "xstest": "domain_superset",
}

benchmark_order = [
    "bbh", "gpqa", "ifeval", "math", "mmlu-pro", "musr",           # reasoning
    "bbq", "do_anything_now", "harmbench","trustllm_jailbreaktrigger", "strongreject", "toxigen", "wildguardtest", "wildjailbreak",  # safety
     "wmdp", "xstest",
]

benchmark_rename = {
    "bbh": "BBH", "gpqa":"GPQA", "mmlu-pro":"MMLU-Pro", "math":"MATH", "musr":"MuSR", "ifeval": "IFEval",          # reasoning
    "harmbench":"HarmBench", "strongreject":"StrongReject", "wildjailbreak":"WildJailbreak", "wildguardtest":"WildGuardTest",  # safety
    "do_anything_now":"Do-Anything-Now", "trustllm_jailbreaktrigger":"JailbreakTrigger", "toxigen":"ToxiGen",
    "bbq":"BBQ", "wmdp":"WMDP", "xstest":"XSTest",
}
legend_ncol = 4
plot_size = 6  # inches, per subplot

base_palette = sns.color_palette("colorblind")  # 10 distinguishable colors

# Per-benchmark color overrides: {benchmark: {category: color}}
palette_overrides = {
    "harmbench": {"copyright": "#f875d0"},
}

# Per-benchmark figure/axes background color overrides: {benchmark: color}
# BACKGROUND_OVERRIDES = {
#     "harmbench": "#faf2e9",
# }
background_overrides = {}

ability_names = {
    "0": "(Safety)",
    "1": "(General\nReasoning)",
}

filter_label_re = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*Filter\s+Ability\s+(\d+)\s*$")

def format_filter_label(label):
    """
    Stack a filter column name onto three lines, e.g.
    "90.0% Filter Ability 0" -> "90% Filter\nAbility 0\n(Safety)".
    Inputs:
        label | str: The column name to format
    Outputs:
        The formatted name
    """
    match = filter_label_re.match(label)
    if match is None:
        return None
    percent = f"{float(match.group(1)):g}"
    ability = match.group(2)
    return f"{percent}% Filter\nAbility {ability}\n{ability_names.get(ability, f'(Ability {ability})')}"


def make_correlation_plots(input_data, output_dir, dim, title, label):
    """
    Make a heatmap given a set of data with correlations and p-values
    Inputs:
        input_data | str: Filename of the csv containing the correlation
        output_dir | str: The folder to save the plots to
        dim | int: The number of the dimension plotted
        title | str: The title to add to the plot
        label | str: The label to add to the saved filename
    Outputs:
        Saves the correlation plot
    """
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })
    df = pd.read_csv(input_data, index_col = "Benchmark")
    df = df.reindex([b for b in benchmark_order if b in df.index])
    df.rename(index=benchmark_rename, inplace=True)

    corr_cols = [c for c in df.columns if f"corr {dim}" in c]
    pval_cols = [c for c in df.columns if f"p {dim}" in c]

    # Keep same order (important)
    corr_cols.sort()
    pval_cols.sort()

    corr_matrix = df[corr_cols].copy()
    pval_matrix = df[pval_cols].copy()

    # Clean column names for display
    corr_matrix.columns = [c.replace(f" corr {dim}", '') for c in corr_cols]
    pval_matrix.columns = [c.replace(f" p {dim}", '') for c in pval_cols]

    # Build annotation matrix with asterisks
    annot_matrix = corr_matrix.copy().astype(str)

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix.iloc[i, j]
            p = pval_matrix.iloc[i, j]
            
            if pd.notna(val) and val != "-":
                p = float(p)
                val = float(val)
                star = '*' if p < 0.01 else ''
                annot_matrix.iloc[i, j] = f"{val:.2f}{star}"
            else:
                annot_matrix.iloc[i, j] = ""
                corr_matrix.iloc[i, j] = None

    # Plot heatmap
    n_rows, n_cols = corr_matrix.shape
    col_width = 1
    plt.figure(figsize=(col_width * n_cols + 2.5, 6))
    ax = sns.heatmap(
        corr_matrix.astype(float),
        annot=annot_matrix,
        annot_kws={'size': 9},
        fmt="",
        cmap="viridis",
        center=0, vmin=-1, vmax=1,
        cbar_kws={'label': 'Correlation'}
    )
    def break_at_keyword(label):
        formatted = format_filter_label(label)
        if formatted is not None:
            return formatted
        return label.replace(' ', '\n', 1)  # e.g. "All Prompts" -> "All\nPrompts"

    ax.set_xticklabels(
        [break_at_keyword(l.get_text()) for l in ax.get_xticklabels()], rotation=0, ha='center')
    # Horizontal divider after 6th row
    ax.hlines(6, *ax.get_xlim(), colors='white', linewidth=3)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{output_dir}corr_{label}.png")



def build_palette(categories, overrides=None):
    """
    Map each category to a unique color. 
    Inputs:
        categories | list: The list of categories to assign colors too
        overrides | dict or None: Any overrides to the base palattes of the graphs
    Outputs:
        The color palatte
    """
    if len(categories) <= len(base_palette):
        colors = base_palette
    else:
        colors = sns.color_palette("husl", n_colors=len(categories))
    palette = {c: colors[i] for i, c in enumerate(categories)}
    if overrides:
        palette.update(overrides)
    return palette


def discrimination_and_difficulty_plots(input_data, output_dir):
    """
    For each benchmark, plot discrimination and difficulty side by side in
    one figure with a single shared legend. 
    Inputs:
        input_data | str: The filename where the IRT parameters are stored
        output_dir | str: The foldername in which to save the results
    """
    df = pd.read_csv(input_data)

    min_val_disc_x = df["Discrimination 0"].min()
    min_val_disc_y = df["Discrimination 1"].min()
    max_val_disc_x = df["Discrimination 0"].max()
    max_val_disc_y = df["Discrimination 1"].max()

    min_val_diff_x = df["Difficulty 0"].min()
    min_val_diff_y = df["Difficulty 1"].min()
    max_val_diff_x = df["Difficulty 0"].max()
    max_val_diff_y = df["Difficulty 1"].max()

    max_categories = max(
        df[df["benchmark"] == b][col].nunique() for b, col in all_benchmarks.items()
    )
    legend_rows = math.ceil(max_categories / legend_ncol)
    legend_height = 0.5 + 0.2 * legend_rows

    for b, col in all_benchmarks.items():
        if b == "bbh":
            leg_font = 10
        else:
            leg_font = 13
        df_filtered = df[df["benchmark"] == b]
        hue_order = sorted(df_filtered[col].dropna().unique())
        palette = build_palette(hue_order, overrides=palette_overrides.get(b))

        fig = plt.figure(figsize=(2 * plot_size, plot_size + legend_height))
        gs = fig.add_gridspec(2, 2, height_ratios=[plot_size, legend_height])
        ax_disc = fig.add_subplot(gs[0, 0])
        ax_diff = fig.add_subplot(gs[0, 1])
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis("off")

        bg = background_overrides.get(b)
        if bg:
            fig.patch.set_facecolor(bg)
            ax_disc.set_facecolor(bg)
            ax_diff.set_facecolor(bg)
            legend_ax.set_facecolor(bg)

        sns.scatterplot(
            data=df_filtered,
            x="Discrimination 0",
            y="Discrimination 1",
            hue=col,
            hue_order=hue_order,
            s=20,
            palette=palette,
            alpha=0.65,
            ax=ax_disc,
            legend=False,
        )
        ax_disc.set_xlim(min_val_disc_x, max_val_disc_x)
        ax_disc.set_ylim(min_val_disc_y, max_val_disc_y)
        ax_disc.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax_disc.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax_disc.set_xlabel("Discrimination 0 (Safety)")
        ax_disc.set_ylabel("Discrimination 1 (General Reasoning)")

        sns.scatterplot(
            data=df_filtered,
            x="Difficulty 0",
            y="Difficulty 1",
            hue=col,
            hue_order=hue_order,
            s=20,
            palette=palette,
            alpha=0.65,
            ax=ax_diff,
            legend=True,
        )
        ax_diff.set_xlim(min_val_diff_x, max_val_diff_x)
        ax_diff.set_ylim(min_val_diff_y, max_val_diff_y)
        ax_diff.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax_diff.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax_diff.set_xlabel("Difficulty 0 (Safety)")
        ax_diff.set_ylabel("Difficulty 1 (General Reasoning)")

        handles, labels = ax_diff.get_legend_handles_labels()
        ax_diff.get_legend().remove()
        if len(labels) > 1:
            legend_ax.legend(
                handles,
                labels,
                loc="center",
                ncol=min(legend_ncol, len(labels)),
                fontsize=leg_font,
                frameon=False,
                markerscale=1.5,
            )

        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"{b}_aug27.png"),
            dpi=300,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)


ability_col = re.compile(r"^(\d+)_d_(\d+)_a$")

def pretty_ability(name):
    """
    Format the ability columns
    Inputs:
        name | str: The column name to format
    Outputs:
        The formatted column name
    """
    m = ability_col.match(str(name).strip())
    if not m:
        return name
    n_dims, index = int(m.group(1)), int(m.group(2))
    return f"{n_dims}D dim{chr(ord('A') + index - 1)}"

def make_correlation_plots_dims(input_data, output_dir, title, label, index_col="benchmark"):
    """
    Make a heatmap given a set of data with correlations and p-values
    Inputs:
        input_data | str: The filename of the correlation csv
        output_dir | str: The foldername to save the plots to
        title | str: The title added to the plot
        label | str: A label added to the filename of the plot
        index_col | str: The name of the index column
    Outputs:


    """
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })
    df = pd.read_csv(input_data, index_col = index_col)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if index_col == "benchmark":
        df = df.reindex([b for b in benchmark_order if b in df.index])
        df.rename(index=benchmark_rename, inplace=True)

    corr_cols = [c for c in df.columns if not f" p" in c]
    pval_cols = [c for c in df.columns if f" p" in c]

    print(corr_cols)
    print(pval_cols)

    # Keep same order (important)
    corr_cols.sort()
    pval_cols.sort()

    corr_matrix = df[corr_cols].copy()
    pval_matrix = df[pval_cols].copy()

    # Clean column names for display: 1_d_1_a -> "1D dimA", 3_d_2_a -> "3D dimB"
    corr_matrix.columns = [pretty_ability(c) for c in corr_cols]
    pval_matrix.columns = [pretty_ability(re.sub(r"\s+p$", "", c)) for c in pval_cols]

    # the abilities x abilities matrix carries the same names down the index
    corr_matrix.index = [pretty_ability(i) for i in corr_matrix.index]
    # Build annotation matrix with asterisks
    annot_matrix = corr_matrix.copy().astype(str)

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix.iloc[i, j]
            p = pval_matrix.iloc[i, j]
            
            if pd.notna(val):
                star = '*' if p < 0.01 else ''
                annot_matrix.iloc[i, j] = f"{val:.2f}{star}"
            else:
                annot_matrix.iloc[i, j] = ""

    # Plot heatmap
    n_rows, n_cols = corr_matrix.shape
    col_width = 1
    plt.figure(figsize=(col_width * n_cols + 2.5, 6))
    ax = sns.heatmap(
        corr_matrix.astype(float),
        annot=annot_matrix,
        fmt="",
        annot_kws={'size': 9},
        cmap="viridis",
        center=0, vmin=-1, vmax=1,
        cbar_kws={'label': 'Correlation'}
)

    def break_at_keyword(label):
        return label.replace('Dim ', 'Dim', 1)  # split at first space

    ax.set_xticklabels(
        [break_at_keyword(l.get_text()) for l in ax.get_xticklabels()], rotation=0, ha='center')
    # Horizontal divider after 6th row
    if index_col == "Benchmark" or index_col == "benchmark":
        ax.hlines(6, *ax.get_xlim(), colors='white', linewidth=3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{output_dir}corr_{label}.png")
