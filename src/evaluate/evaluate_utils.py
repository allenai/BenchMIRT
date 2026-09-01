"""
This file contains helper functions for running and analyzing IRT functions
Notably, an order of running may be: 
    create_question_splits()
    format_questions()
    make_model_metadata()
    run_irt() [from irt_setup.py]
    make_item_statistics()
    make_model_statistics()
    

"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from py_irt.io import read_json

def make_scatterplot(x_axis, y_axis, hue, title, filename, s=2):
    """
    Boilerplate for scatterplots.

    Inputs: 
        x_axis | pd.Series: The data series to be plotted on the x-axis
        y_axis | pd.Series: The data series to be plotted on the y-axis
        hue | pd.Series: The data series controlling the color of the data
        title | str: The plot title
        filename | str: The filepath to save the figure
        s | int: The size of the data markers
    Outputs:
        Save the figure to filename
    """
    ax = sns.scatterplot(x=x_axis, y=y_axis, hue=hue, s=s)
    plt.legend(markerscale=3)
    sns.move_legend(
        ax, 
        "upper left",
        bbox_to_anchor=(1, 1),
    )
    if s == 2:
        plt.tight_layout()
    plt.xlabel(x_axis.name)
    plt.ylabel(y_axis.name)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()
    plt.close("all")


def create_question_splits(capability_file, safety_file, foldername):
    """
    Takes the full set of capability results and safety results, joins them, and saves the filtered versions

    Inputs:
        capability_file | str: The location of the processed capability results
        safety_file | str: The location of the processed safety results
        foldername | str: The location to save the outputs
    Outputs:
        Saves two csvs:
            - The combined version of the data in foldername as prompts_all.csv
            - The filtered version of the data with all low variance items removed

    """

    df_capability = pd.read_csv(capability_file, index_col = "question_id")
    df_acc_capability = df_capability.drop(columns=["question", "subset"])

    df_safety = pd.read_csv(safety_file, index_col = "question_id")
    
    # Pull the column names that have model results
    x = [f for f in df_capability.columns if f.startswith("acc")]

    df_safety = df_safety[x]
    df_acc_capability = df_acc_capability[x]

    # Create the and save df with all items
    stacked = pd.concat([df_acc_capability, df_safety])
    
    stacked.to_csv(f"{foldername}prompts_all.csv")

    # Filter the prompts based on accuracy
    acc = stacked.mean(axis=1)
    filtered = stacked[(acc > 0.05) & (acc < 0.95)]
    filtered.to_csv(f"{foldername}prompts_filtered.csv")


def read_table(filename, metadata):
    """
    Process the loss output of the IRT trainer

    Inputs:
        filename | str: The location of the txt file with the loss output
        metadata | list: The hyperparameters from the run we are pulling
    
    Outputs:
        A dataframe with the loss information

    """

    rows = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            
            # Keep only lines that contain actual data rows
            if line.startswith("│") and "Epoch" not in line:
                parts = [p.strip() for p in line.split("│")[1:-1]]
                
                # Skip empty/malformed rows
                if len(parts) == 4:
                    parts = parts + metadata
                    rows.append(parts)
            
            if "┏━" in line:
                rows = []

    # Create DataFrame
    df = pd.DataFrame(rows, columns=["Epoch", "Loss", "Best Loss", "New LR", "Model", "Starting LR", "LR Decay", "Total Epochs", "run_id"])

    # Convert numeric columns
    df = df.astype({
        "Epoch": int,
        "Loss": float,
        "Best Loss": float,
        "New LR": float,
        "Model": str,
        "Starting LR": float,
        "LR Decay": float,
        "Total Epochs": int,
        "run_id": str
    })

    return df



def graph_loss(loss_data, foldername, file=True):
    """
    Create a graph of the loss from a IRT training run

    Inputs:
        loss_data | str or pd.DataFrame: Either the string with the location of the data to graph, or the data to graph
        foldername | str: The location to save the graphs
        file | bool: If true, loss_data is a file where the data is stored, if false it is a dataframe
    Outputs:
        A graph of the loss over the IRT run
    """
    # Load the data
    if file:
        df = pd.read_csv(loss_data)
    else:
        df = loss_data
    
    # Graph each different run
    for v in df["run_id"].unique():

        #Set axes
        graph_df = df[df["run_id"] == v]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()   
        ax2.set_yscale('log')
        ax1.set_yscale('log')

        # Plot loss on the left axis and lr on the right
        ax1.plot(graph_df["Epoch"], graph_df["Loss"], label="Loss", color='blue', linestyle='-')
        ax1.plot(graph_df["Epoch"], graph_df["Best Loss"], label="Best Loss", color='green', linestyle='-')
        ax2.plot(graph_df["Epoch"], graph_df["New LR"], label="Learning Rate", color='yellow', linestyle='--')
        
        # Set labels
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Learning Rate")
        plt.title(f"Loss and LR for {v} Model")

        # Create a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.savefig(f"{foldername}loss_{v}.png")
        plt.clf()
        plt.close('all')



def format_questions(filename, output_path, item_metadata, bench=None, category=None, file=True, df=None):
    """
    Read the create_question_splits csv in and transform it into the format necessary for the irt model

    Inputs:
        filename | str or None: If file is True, the path to the data to format.
        output_path | str: The location to save the results
        item_metadata | str: The location of the csv with the item_metadata
        bench | str or None: If not none, the name of the benchmark to filter
        category | str or None: If not none, the name of the category to filter
        file | bool: If true, read the data in from filename. If false, read the data in from df
        df | pd.DataFrame or None: Contains the data to format if file is False
    
    Outputs:
        A correctly formatted jsonl at output_path

    """
    if file:
        df = pd.read_csv(filename, index_col="question_id")
    
    item_metadata = pd.read_csv(item_metadata, index_col="question_id")
    output = []
    for c in df.columns:
        if c.startswith("acc_norm"):
            model_name = c[9:]
            responses = {}
            for i, row in df.iterrows():
                if bench is not None:
                    if item_metadata.loc[i, "benchmark"] == bench:
                        responses[i] = int(row[c])
                    else:
                        pass
                elif category is not None:
                    if item_metadata.loc[i, "category"] == category:
                        responses[i] = int(row[c])
                    else:
                        pass
                else:
                    responses[i] = int(row[c])
            output.append({"subject_id": model_name, "responses": responses})
    
    with open(output_path, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")
    return  



def make_model_statistics(irt_results, model_metadata, foldername, dims, tag=""):
    """
    Collect the results of an IRT model into a csv for visualization of models

    Inputs:
        irt_results | str: The path to a json file with the parameters from an IRT run
        model_metadata | str: The path to a csv file with the metadata about models
        foldername | str: The location to save the result to
        dims | int: The number of dimensions used in the IRT model, always 1 if it was a single dim run
        tag | str: Optional argument to add to the file name to distinguish multiple runs

    Outputs:
        A csv with the model statistics saved to foldername

    """

    df_model_metadata = pd.read_csv(model_metadata, index_col="model")
    irt = read_json(irt_results)
    
    model_abilities = []
    for model_idx in irt["subject_ids"].keys():
        m = {
            "model": irt["subject_ids"][model_idx]
        }
        if dims == 1:
            m["Ability 0"] = irt["ability"][int(model_idx)]
        else:
            for i in range(dims):
                m["Ability " + str(i)] = irt["ability"][int(model_idx)][i]
        model_abilities.append(m)
    
    df_model_abilities = pd.DataFrame(model_abilities).set_index("model")
    df_model = df_model_metadata.merge(df_model_abilities, left_index=True, right_index=True)
    df_model.to_csv(f"{foldername}model_statistics_{tag}.csv")

    return df_model



def make_item_statistics(irt_results, item_metadata, foldername, dims, tag=""):
    """
    Collect the results of an IRT model into a csv for visualization of items

    Inputs:
        irt_results | str: The path to a json file with the parameters from an IRT run
        item_metadata | str: The path to a csv file with the metadata about prompts
        foldername | str: The location to save the result to
        dims | int: The number of dimensions used in the IRT model, always 1 if it was a single dim run
        tag | str: Optional argument to add to the file name to distinguish multiple runs

    Outputs:
        A csv with the item statistics saved to foldername
    """
    df_item_metadata = pd.read_csv(item_metadata, index_col="question_id")
    irt = read_json(irt_results)
    if irt_results.endswith("rotated.json"):
        ids = read_json(irt_results[:-13] + ".json")
        irt["ability"] = irt.pop("ability_rot")
        irt["disc"] = irt.pop("disc_rot")
        irt["diff"] = irt.pop("diff_rot")
        irt["item_ids"] = ids.pop("item_ids")
        irt["subject_ids"] = ids.pop("subject_ids")
        irt["item_ids"] = {int(i): j for i, j in irt["item_ids"].items()}
        irt["subject_ids"] = {int(i): j for i, j in irt["subject_ids"].items()}

    item_statistics = []
    for item_idx in irt["item_ids"].keys():
        m = {
            "question_id": irt["item_ids"][item_idx]
        }
        if dims == 1:
            m["Difficulty 0"] = irt["diff"][int(item_idx)]
            m["Discrimination 0"] = irt["disc"][int(item_idx)]
        else:
            for i in range(dims):
                m["Difficulty " + str(i)] = irt["diff"][int(item_idx)][i]
                m["Discrimination " + str(i)] = irt["disc"][int(item_idx)][i]
        item_statistics.append(m)
    
    df_item_statistics = pd.DataFrame(item_statistics).set_index("question_id")
    df_item = df_item_metadata.merge(df_item_statistics, left_index=True, right_index=True)
    df_item.to_csv(f"{foldername}item_statistics_{tag}.csv")

    return df_item



def make_model_metadata(file, item_metadata, model_static_metadata, filter_type, foldername):
    """
    Pull static metadata info about models and combine it with dynamic results from the filtered data

    Inputs:
        file | str: The location of the data, in the format created by create_question_splits
        item_metadata | str: The location of the prompt metadata
        model_static_metadata | str: The location of the static model information
        filter_type | str: The type of filter applied to the items in file, used only for naming
        foldername | str: The location to save the result

    Outputs:
        Saves the metadata to a csv in foldername

    """
    df = pd.read_csv(file, index_col="question_id")
    item = pd.read_csv(item_metadata, index_col="question_id")
    df = df.merge(item[[ "benchmark", "category", "domain_superset", "subcategory_one"]], how="left", left_index=True, right_index=True)
    
    # Prepare to calculate averages for different groups
    df_total = df.drop(columns=["benchmark", "category", "domain_superset", "subcategory_one"])
    df_benchmark = df.drop(columns=["category", "domain_superset", "subcategory_one"])
    df_category = df.drop(columns=["benchmark", "domain_superset", "subcategory_one"])
    df_domain = df.drop(columns=["benchmark", "category", "subcategory_one"])
    df_subcat = df.drop(columns=["benchmark", "category", "domain_superset"])
    df_select_safety = df[(df["category"] == "Safety") & (df["benchmark"] != "wmdp") & (df["benchmark"] != "bbq")]
    
    # Calculate averages
    df_benchmark = df_benchmark.groupby("benchmark").mean()
    df_category = df_category.groupby("category").mean()
    df_domain = df_domain.groupby("domain_superset").mean()
    df_subcat = df_subcat.groupby("subcategory_one").mean()
    select_safety = df_select_safety.drop(columns=["benchmark", "category", "domain_superset", "subcategory_one"]).mean()
    df_total = df_total.mean()

    model_info = pd.read_csv(model_static_metadata)
    model_df = []

    # Add static metadata and dynamic metadata together
    for i, row in model_info.iterrows():
        col_name = "acc_norm_" + row["Model"]
        m = {
            "model": row["Model"],
            "family": row["family"],
            "size": row["size"],
            "reasoning": row["reasoning"],
            "date": row["date"],
            "total_average": df_total[col_name],
            "reasoning_average": df_category.loc["General Reasoning", col_name],
            "safety_average": df_category.loc["Safety", col_name],
            "select_safety_average": select_safety[col_name]
        }

        # Add benchmark and domain averages
        for i, row in df_benchmark.iterrows():
            m[i + "_average"] = row[col_name]
        for i, row in df_domain.iterrows():
            m["domain_" + i + "_average"] = row[col_name]
        for i, row in df_subcat.iterrows():
            m["subcat_" + i + "_average"] = row[col_name]
        model_df.append(m)

    m_df = pd.DataFrame(model_df)
    m_df = m_df.set_index("model")
    m_df.to_csv(f"{foldername}model_metadata_{filter_type}.csv")