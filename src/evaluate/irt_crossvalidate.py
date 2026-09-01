"""
This file contains the code to run the accuracy cross-validation comparing MIRT against baselines
"""

import json

import dataframe_image as dfi
import numpy as np
import pandas as pd

from py_irt.dataset import Dataset
from py_irt.config import IrtConfig
from py_irt.io import write_json
from py_irt.training import IrtModelTrainer
from sklearn.model_selection import KFold
import fire

from evaluate_constants import all_benchmarks, heldout_data_baselines_to_run

pd.set_option('styler.render.max_rows', None)
pd.set_option('display.max_rows', None)



def calculate_mae(df, predict=None, return_value=False):
    """
    Calculate the total Mean Average Error for a given set of predicted
    and actual values. By default return just the total. Otherwise, return
    with a subject-by-item df

    Inputs: 
        df | pd.DataFrame: DataFrame with columns subject_id, item_id, actual. If it does not have predicted, then predict must not be None
        predict | pd.DataFrame or None: Optional DataFrame with columns subject_id, item_id, and predicted
        return_value | bool: If true, then return a dataframe with item by item calculation in addition to the total number
    
    Outputs:
        The MAE for all input data, and if return_value the dataframe used to calculate that
    """
    if predict is not None:
        df = df.merge(predict, on=["subject_id", "item_id"])
    
    df["mae"] = abs(df["actual"] - df["predicted"])

    if not return_value:
        return df["mae"].mean()
    
    return df["mae"].mean(), df[["subject_id", "item_id", "benchmark", "mae"]]



def calculate_accuracy(df, predict=None, return_value=False):
    """
    Calculate the total accuracy for a given set of predicted
    and actual values. By default return just the total. Otherwise, return
    with a subject-by-item df

    Inputs: 
        df | pd.DataFrame: DataFrame with columns subject_id, item_id, actual. If it does not have predicted, then predict must not be None
        predict | pd.DataFrame or None: Optional DataFrame with columns subject_id, item_id, and predicted
        return_value | bool: If true, then return a dataframe with item by item calculation in addition to the total number
    
    Outputs:
        The average accuracy for all input data, and if return_value the dataframe used to calculate that
    """
    if predict is not None:
        df = df.merge(predict, on=["subject_id", "item_id"])
    
    df["accuracy"] = np.round(df["predicted"]) == df["actual"]

    if not return_value:
        return df["accuracy"].mean()
    
    return df["accuracy"].mean(), df[["subject_id", "item_id", "benchmark", "accuracy"]]



def calculate_ranking(df, predict=None, return_value=False):
    """
    Calculate the total correlation between model rankings for a given set of predicted
    and actual values. By default return just the total. Otherwise, return
    with a model by benchmark df with rankings of actual and predicted for each model on each benchmarks

    Inputs: 
        df | pd.DataFrame: DataFrame with columns subject_id, item_id, actual. If it does not have predicted, then predict must not be None
        predict | pd.DataFrame or None: Optional DataFrame with columns subject_id, item_id, and predicted
        return_value | bool: If true, then return a dataframe with item by item calculation in addition to the total number
    
    Outputs:
        The spearman correlation between ranks for all input models, and if return_value the dataframe used to calculate that
    
    """

    if predict is not None:
        df = df.merge(predict, on=["subject_id", "item_id"])
    df_rank = df.groupby(["subject_id"])[["actual", "predicted"]].mean().reset_index()
    df_rank["actual_rank"] = df_rank["actual"].rank(method="min", ascending=False)
    df_rank["predicted_rank"] = df_rank["predicted"].rank(method="min", ascending=False)
    spearman_corr = df_rank["actual_rank"].corr(df_rank["predicted_rank"], method="spearman")


    if not return_value: 
        return spearman_corr
    
    df_rank_bench = df.groupby(["subject_id", "benchmark"])[["actual", "predicted"]].mean().reset_index()
    df_rank_bench["actual_rank"] = df_rank_bench.groupby('benchmark')["actual"].rank(method="min", ascending=False)
    df_rank_bench["predicted_rank"] = df_rank_bench.groupby('benchmark')["predicted"].rank(method="min", ascending=False)
    return spearman_corr, df_rank_bench[["subject_id", "benchmark", "actual_rank", "predicted_rank"]]



def subject_baseline_table(df, agg_column, agg_name, output_file):
    """
    Create a table with the metric results of one of the baselines

    Inputs:
        df | pd.DataFrame: Prediction values for all items
        agg_column | pd.DataFrame or None: The results of running the metric
        agg_name | str: The name of the metric type
        output_file | str: The name of the file to save the table to
    
    Output:
        A benchmark table at output_file
    """
    print("calling table", agg_name)
    if agg_column is not None:
        df = df.merge(agg_column, on=['subject_id', 'item_id', 'benchmark'])
    pivot = pd.pivot_table(df, index="benchmark", columns="subject_id", values=agg_name, aggfunc="mean").apply(pd.to_numeric, errors="coerce")
    pivot.to_csv(output_file[:-3] + "csv")



def train_model_average_total(dataset):
    """
    Returns a dictionary of subject_id:total_average pairs

    Used to predict unseen items as the average of each subject_id's performance on training data

    """
    df = pd.DataFrame(dataset)

    avg_df = df.groupby("subject_id")["actual"].mean()

    scores = {}
    for i, row in avg_df.items():
        scores[i] = row
    
    return scores



def predict_model_average_total(data, train):
    """
    Return a dataframe of predictions using the subject_id's average score for all items
    """
    for row in data:
        row["predicted"] = train[row["subject_id"]]
    
    df = pd.DataFrame(data)

    return df[["subject_id", "item_id", "predicted"]]



def train_model_average_category(dataset):
    """
    Returns a dictionary of subject_id:{"category_1":score, "category_2":score}

    Used to predict unseen items as the average of each subject_id's performance on training data in that category
    """
    df = pd.DataFrame(dataset)

    avg_df = df.groupby(["subject_id", "category"])["actual"].mean()
    scores = {}
    for i, row in avg_df.items():
        if i[0] in scores.keys():
            scores[i[0]][i[1]] = row
        else:
            scores[i[0]] = {i[1]: row}
    
    return scores



def predict_model_average_category(data, train):
    """
    Returns a dataframe of predictions using the subject_id's average score on the category of each item
    """
    for row in data:
        row["predicted"] = train[row["subject_id"]][row["category"]]
    
    df = pd.DataFrame(data)

    return df[["subject_id", "item_id", "predicted"]]



def train_model_average_benchmark(dataset):
    """
    Returns a dictionary of subject_id:{benchmark:score...benchmark:score}

    Used to predict unseen items as the average of each subject_id's performance on training data in that benchmark
    """
    df = pd.DataFrame(dataset)

    avg_df = df.groupby(["subject_id", "benchmark"])["actual"].mean()

    scores = {}
    for i, row in avg_df.items():
        if i[0] in scores.keys():
            scores[i[0]][i[1]] = row
        else:
            scores[i[0]] = {i[1]: row}
    
    return scores 


def predict_model_average_benchmark(data, train):
    """
    Returns a dataframe of predictions using the subject_id's average score on the benchmark of each item
    """
    for row in data:
        row["predicted"] = train[row["subject_id"]][row["benchmark"]]
    
    df = pd.DataFrame(data)

    return df[["subject_id", "item_id", "predicted"]]



def multidim_predict(model_params, subject, item, subject_ids, item_ids):
    """
    Return a single prediction from a multidimensional 2pl model
    """
    abilities = np.array(model_params["ability"][subject_ids[subject]])
    diffs = np.array(model_params["diff"][item_ids[item]])
    discs = np.array(model_params["disc"][item_ids[item]])
    intermediate_1 = np.subtract(abilities, diffs)
    intermediate_2 = -np.dot(discs.T, (intermediate_1))
    intermediate_3 = 1 + np.exp(intermediate_2)
    pred = 1 / intermediate_3
    return pred



def singledim_predict(model_params, subject, item, subject_ids, item_ids):
    """
    Return a single prediction from a single dimensional 2pl model
    """
    abilities = model_params["ability"][subject_ids[subject]]
    diffs = model_params["diff"][item_ids[item]]
    discs = model_params["disc"][item_ids[item]]
    return 1 / (1 + np.exp(-(discs * (abilities-diffs))))



def irt_format_data(data, parameters, baseline_name, division):
    """
    Accepts a list of dictionaries with keys: subject_id, benchmark, category, item_id, actual
    Returns an IRT dataset object
    """

    df = pd.DataFrame(data)
    df = df.pivot(index="subject_id", columns="item_id", values="actual")
    df.to_csv(f"{parameters["foldername"]}_best_parameters_{baseline_name}_{parameters["dims"]}_{division}_training.csv")
    dataset = Dataset.from_pandas(df)

    return dataset



def irt_train(training_data, parameters, baseline_name, division=""):
    """
    Helper function for training irt models
    """
    irt_config = IrtConfig(
        model_type=parameters["model_type"],
        priors=parameters["priors"],
        epochs=parameters["epochs"],
        dims=parameters["dims"],
        lr=parameters["lr"],
        lr_decay=parameters["lr_decay"],
        initializers=parameters["initializers"]
    )

    # Try training multiple times and catch errors if the process crashes
    count = 0
    trained = False
    params = None 

    while count < 50 and trained == False:

        try:
            trainer = IrtModelTrainer(
                dataset=training_data,
                data_path=parameters["data_path"],
                config=irt_config,
                verbose=True
            )

            trainer.train(epochs=parameters["epochs"], device="cuda")
            params = trainer.best_params
            write_json(f"{parameters["foldername"]}_best_parameters_{baseline_name}_{parameters["dims"]}_{division}.json", trainer.best_params)
            trained = True 
        
        except:
            count += 1
    
    if params is not None:
        return params

    print(f"Failed to train the {baseline_name}")
    return None



def train_multi_dim_irt_total(dataset, baseline_name, parameters, fold=""):
    """
    Train a multidimensional irt model on the all training data and return the trained model parameters
    """

    training_data = irt_format_data(dataset, parameters, baseline_name, fold)

    parameters["model_type"] = "multidim_2pl"

    params = irt_train(training_data, parameters, baseline_name, fold)

    return params



def predict_multi_dim_irt_total_heldout(data, train):
    """
    Predict model outputs using a trained multidim model
    Return dataframe with subject_id, item_id, predicted
    """
    subject_ids = {v: k for k, v in train["subject_ids"].items()}
    item_ids = {v: k for k, v in train["item_ids"].items()}
    for row in data:
        row["predicted"] = multidim_predict(train, row["subject_id"], row["item_id"], subject_ids, item_ids)
    
    df = pd.DataFrame(data)

    return df[["subject_id", "item_id", "predicted"]]



def train_single_dim_irt_total(dataset, baseline_name, parameters, fold):
    """
    Train a single irt model on all the input data and return the trained model parameters
    """

    training_data = irt_format_data(dataset, parameters, baseline_name, fold)

    parameters["model_type"] = "2pl"

    params = irt_train(training_data, parameters, baseline_name, fold)

    return params



def predict_single_dim_irt_total_heldout(data, train):
    """
    Predict held out data using trained irt model
    Train is a set of params
    """
    subject_ids = {v: k for k, v in train["subject_ids"].items()}
    item_ids = {v: k for k, v in train["item_ids"].items()}

    for row in data:
        row["predicted"] = singledim_predict(train, row["subject_id"], row["item_id"], subject_ids, item_ids)
    
    df = pd.DataFrame(data)

    return df[["subject_id", "item_id", "predicted"]]


def train_single_dim_irt_category(dataset, baseline_name, parameters, fold):
    """
    Train two single dimensional irt models, one on all category 1 data and one on all category 2 data
    Return a dictionary in the form {"category_1": params, "category_2": params}
    """

    param_dict = {}

    for cat in list({d["category"] for d in dataset}):
        d = [item for item in dataset if item["category"] == cat]
        training_data = irt_format_data(d,  parameters, f"{baseline_name}_{cat}", fold)

        parameters["model_type"] = "2pl"

        params = irt_train(training_data, parameters, f"{baseline_name}_{cat}", fold)
        
        param_dict[cat] = params 
    
    return param_dict
        


def predict_single_dim_irt_category_heldout(data, train):
    """
    Predict held out data using two trained single dimensional irt models
    Train is a dictionary of params
    """
    subject_ids = {}
    item_ids = {}
    for c in list(train.keys()):
        subject_ids[c] = {v: k for k, v in train[c]["subject_ids"].items()}
        item_ids[c] = {v: k for k, v in train[c]["item_ids"].items()}

    for row in data:
        row["predicted"] = singledim_predict(train[row["category"]], row["subject_id"], row["item_id"], subject_ids[row["category"]], item_ids[row["category"]])
    
    df = pd.DataFrame(data)

    return df[["subject_id", "item_id", "predicted"]]


def train_model(dataset, baseline_name, parameters, fold):
    """
    Helper function that calls the correct training method for each baseline
    """
  
    if parameters["baseline_type"] == "Model_Average_Total":
        return train_model_average_total(dataset)
    elif parameters["baseline_type"] == "Model_Average_Category":
        return train_model_average_category(dataset)
    elif parameters["baseline_type"] == "Model_Average_Benchmark":
        return train_model_average_benchmark(dataset)
    elif parameters["baseline_type"].startswith("Multi_Dim_IRT_Total"):
        return train_multi_dim_irt_total(dataset, baseline_name, parameters, fold)
    elif parameters["baseline_type"].startswith("Single_Dim_IRT_Total"):
        return train_single_dim_irt_total(dataset, baseline_name, parameters, fold)
    elif parameters["baseline_type"].startswith("Single_Dim_IRT_Category"):
        return train_single_dim_irt_category(dataset, baseline_name, parameters, fold)
    else:
        print("Baseline", baseline_name, "is not currently supported")


def predict_model(data, baseline_name, parameters, train):
    """
    Helper function that calls the correct prediction method for each baseline
    """
    if parameters["baseline_type"]  == "Model_Average_Total":
        return predict_model_average_total(data, train)
    elif parameters["baseline_type"] == "Model_Average_Category":
        return predict_model_average_category(data, train)
    elif parameters["baseline_type"] == "Model_Average_Benchmark":
        return predict_model_average_benchmark(data, train)
    elif parameters["baseline_type"] == "Multi_Dim_IRT_Total_Heldout":
        return predict_multi_dim_irt_total_heldout(data, train)
    elif parameters["baseline_type"] == "Single_Dim_IRT_Total_Heldout":
        return predict_single_dim_irt_total_heldout(data, train)
    elif parameters["baseline_type"] == "Single_Dim_IRT_Category_Heldout":
        return predict_single_dim_irt_category_heldout(data, train)
    else:
        print("Baseline", baseline_name, "is not currently supported")


def run_heldout_split(input_data, output_dir, item_metadata, baselines_to_run=list(heldout_data_baselines_to_run.keys())):
    """
    Run an analysis, holding out 10% of data at a time
    """
    # Pull item - benchmark data
    item_to_benchmark = {}
    metadata = pd.read_csv(item_metadata)
    for i, row in metadata.iterrows():
        item_to_benchmark[row["question_id"]] = [row["benchmark"], row["category"], row["question_text"]]
    print(len(item_to_benchmark))
    # Load in the full set of data
    data = []
    ids = []
    with open(input_data) as f:
        i =0
        for line in f:
            print(i)
            submission = json.loads(line)
            model_id = submission["subject_id"]
            for example_id in submission["responses"].keys():
                data.append({
                    "subject_id": model_id,
                    "benchmark": item_to_benchmark[example_id][0],
                    "category": item_to_benchmark[example_id][1],
                    "item_id": example_id,
                    "actual": submission["responses"][example_id]
                })
                ids.append((model_id, example_id))
            i += 1
    print(len(ids))
    d = pd.DataFrame(data)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(d)):
        training_d = d.iloc[train_idx]
        test_d = d.iloc[val_idx]
        print("Starting fold", fold, "with length training size", len(training_d), "and length testing size", len(test_d))
        training_d.to_csv(f"{output_dir}fold_{fold}_training_d.csv")
        test_d.to_csv(f"{output_dir}fold_{fold}_test_d.csv")
        test_set = []
        training_set = []
        actual_set = []
        # print(len(data))
        # d = pd.DataFrame(data)
        for i, row in training_d.iterrows():
            training_set.append({
                    "subject_id": row["subject_id"],
                    "category": row["category"],
                    "benchmark": row["benchmark"],
                    "item_id": row["item_id"],
                    "actual": row["actual"]
                })
        for i, row in test_d.iterrows():
            test_set.append({
                    "subject_id": row["subject_id"],
                    "category": row["category"],
                    "benchmark": row["benchmark"],
                    "item_id": row["item_id"],
                })
            actual_set.append({
                    "subject_id": row["subject_id"],
                    "benchmark": row["benchmark"],
                    "item_id": row["item_id"],
                    "actual": row["actual"]
                })


        actual_set = pd.DataFrame(actual_set)

        # Create the final table
        index_scores = list(all_benchmarks.keys()) + ["micro-accuracy", "macro-accuracy"]
        final_scores = pd.DataFrame(index_scores, columns=["benchmark"])
        # For each baseline
        for baseline in baselines_to_run:
            print("Running", baseline)
            heldout_data_baselines_to_run[baseline]["data_path"] = input_data 
            heldout_data_baselines_to_run[baseline]["foldername"] = output_dir 

            baseline_scores = []

            # Train on the all but
            train = train_model(training_set, baseline, heldout_data_baselines_to_run[baseline], fold)
            # print(baseline, train)

            # Test on the only
            prediction = predict_model(test_set, baseline, heldout_data_baselines_to_run[baseline], train)
            # print(baseline, prediction)

            # Score with MAE
            mae_micro, mae_values = calculate_mae(actual_set, predict=prediction, return_value=True)
            # print(mae_micro, mae_values)

            # Score with accuracy
            accuracy_micro, accuracy_values = calculate_accuracy(actual_set, predict=prediction, return_value=True)
            # print(accuracy_micro, accuracy_values)

            # Score with ranking
            ranking_micro, ranking_values = calculate_ranking(actual_set, predict=prediction, return_value=True)
            # print(ranking_micro, ranking_values)

            # concat to a dataframe predictions
            # AGGREGATE HERE
            for bench in all_benchmarks.keys():
                df_rank = ranking_values[ranking_values["benchmark"] == bench]
                spearman_corr = df_rank["actual_rank"].corr(df_rank["predicted_rank"], method="spearman")
                baseline_scores.append({
                    "benchmark": bench,
                    f"{baseline}_mae": mae_values[mae_values["benchmark"] == bench]["mae"].mean(),
                    f"{baseline}_accuracy": accuracy_values[accuracy_values["benchmark"] == bench]["accuracy"].mean(),
                    f"{baseline}_ranking": spearman_corr,
                })
            
            # Save a predictions csv
            baseline_predictions = prediction.merge(actual_set, on=["subject_id", "item_id"])
            baseline_predictions.to_csv(f"{output_dir}_predictions_{baseline}fold_{fold}_.csv")

            # Calculate the macro totals 
            mae_average = sum(b[f"{baseline}_mae"] for b in baseline_scores) / len(baseline_scores)
            accuracy_average = sum(b[f"{baseline}_accuracy"] for b in baseline_scores) / len(baseline_scores)
            ranking_average = sum(b[f"{baseline}_ranking"] for b in baseline_scores) / len(baseline_scores)

            baseline_scores.append({
                "benchmark": "macro-accuracy",
                f"{baseline}_mae": mae_average,
                f"{baseline}_accuracy": accuracy_average,
                f"{baseline}_ranking": ranking_average,
            })

            # Calculate the micro totals 
            baseline_scores.append({
                "benchmark": "micro-accuracy",
                f"{baseline}_mae": mae_micro,
                f"{baseline}_accuracy": accuracy_micro,
                f"{baseline}_ranking": ranking_micro,
            })
            

            print(baseline_scores)
            # Add a column to a dataframe with benchmark scores, average over benchmarks, and average over averages
            baseline_df = pd.DataFrame(baseline_scores)
            final_scores = final_scores.merge(baseline_df, on='benchmark')

            # Make a benchmark - by - models accuracy table
            subject_baseline_table(baseline_predictions, mae_values, "mae", f"{output_dir}_mae_{baseline}_fold_{fold}.png")
            subject_baseline_table(baseline_predictions, accuracy_values, "accuracy", f"{output_dir}_accuracy_{baseline}_fold_{fold}.png")
        
        final_scores = final_scores.set_index("benchmark")
        # Make the final large table
        final_scores_stylized = (
            final_scores.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None).set_table_styles([
                    {"selector": "th", "props": [("font-weight", "bold")]},
                    {"selector": "td", "props": [("text-align", "center")]},
                    {"selector": "table", "props": [("border-collapse", "collapse")]},
                    {"selector": "td, th", "props": [("border", "1px solid #ccc")]}
                ]).set_properties(
                    **{'font-weight': 'bold'}, subset=pd.IndexSlice['micro-accuracy', :]).set_properties(
                        **{'font-weight': 'bold'}, subset=pd.IndexSlice['macro-accuracy', :])
                        )
        try:
            dfi.export(final_scores_stylized, f"{output_dir}_fold_{fold}_all_scores.png", table_conversion='matplotlib')
            final_scores.to_csv(f"{output_dir}_fold_{fold}_all_scores.csv")
        except:
            print("could not make table")
            final_scores.to_csv(f"{output_dir}_fold_{fold}_all_scores.csv")
        # Make a smaller table for each accuracy measurement
        for m in ["mae", "accuracy", "ranking"]:
            cols = [c for c in final_scores if c.endswith(m)]
            subset_df = final_scores[cols]
            subset_stylized = (
            subset_df.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None).set_table_styles([
                    {"selector": "th", "props": [("font-weight", "bold")]},
                    {"selector": "td", "props": [("text-align", "center")]},
                    {"selector": "table", "props": [("border-collapse", "collapse")]},
                    {"selector": "td, th", "props": [("border", "1px solid #ccc")]}
                ]).set_properties(
                    **{
                        "font-weight": "bold",
                        "background-color": "#f5f5f5",
                        "border-top": "2px solid black"
                    },
                    subset=pd.IndexSlice[['micro-accuracy', 'macro-accuracy'], :]
                )
                        )
            try:
                dfi.export(subset_stylized, f"{output_dir}_{m}_fold_{fold}_scores.png", table_conversion='matplotlib')
            except:
                print("could not make table")
                subset_df.to_csv(f"{output_dir}_{m}_fold_{fold}_scores.csv")

def main(input_data, output_dir, item_metadata, baselines_to_run = None):
    """
    Main function to run testing
    """
    if baselines_to_run is None:
        baselines_to_run=list(heldout_data_baselines_to_run.keys())
    run_heldout_split(input_data, output_dir, item_metadata, baselines_to_run)

if __name__ == "__main__":
    fire.Fire(main)
