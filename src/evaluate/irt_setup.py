"""
This file contains the code to run and visualize a single IRT or MIRT run
Be aware that due to numeric instability an IRT run may require several restarts in order
to run without crashing
"""
import dataframe_image as dfi
import pandas as pd

from py_irt.config import IrtConfig
from py_irt.io import write_json
from py_irt.training import IrtModelTrainer
from scipy.stats import pearsonr

from evaluate_utils import make_item_statistics, make_model_statistics, make_scatterplot

pd.set_option("styler.render.max_rows", 200)
pd.set_option("styler.render.max_columns", 200)
pd.set_option("styler.render.max_elements", 1_000_000)
 

def run_irt(model_type, data_path, foldername, tag, epochs=5000, priors="vague", dims=2, lr=0.2, lr_decay=0.9999, device="cuda", initializers=[]):
    """
    Run a single instance of an IRT model

    Input:
        model_type | str: The IRT model type to use
        data_path | str: The path to the data, in IRT model form (output of format_questions())
        foldername | str: The folder in which to store results
        tag | str: An extra label to append to the file
        epochs | int: The number of epochs to train the model to
        priors | str: The priors for the IRTConfig
        dims | int: The number of dimensions for the IRT model, always 1 for a single dimensional model
        lr | float: The learning rate for the IRT models
        lr_decay | float: The rate at which the learning rate decays
        device | str: Either cuda or cpu, depending on if there is a gpu available
        initializers | list: The initializers for the IRT Config

    Output:
        Saves the best parameters from the run and the final parameters from the run
        All downstream code uses the best parameters

    """

    irt_config = IrtConfig(
        model_type=model_type,
        priors=priors,
        epochs=epochs,
        dims=dims,
        lr=lr,
        lr_decay=lr_decay,
        initializers=initializers
    )

    trainer = IrtModelTrainer(
        data_path=data_path,
        config=irt_config,
        verbose=True
    )

    trainer.train(epochs=epochs, device=device)
    trainer.save(foldername + f"/parameters_{model_type}_{dims}_{lr}_{tag}.json")
    write_json(foldername + f"/best_parameters_{model_type}_{dims}_{lr}_{tag}.json", trainer.best_params)



def graph_irt_results(irt_results, item_metadata, model_metadata, dims, foldername, s=15):
    """
    Visualize the results of an IRT run

    Inputs:
        irt_results | str: A json file with the parameters from an IRT run
        item_metadata | str: The location of the prompt metadata file
        model_metadata | str: The location of the model metadata file
        dims | int: The number of dimensions the IRT model was run on
        foldername | str: The location to save the results to 
        s | int: The size of dots to graph with
    
    Outputs:
        A set of graphs and styled tables in foldername. 
        Note that foldername should exist and have a subdirectory named "benchmarks"

    """
    
    df_item = make_item_statistics(irt_results, item_metadata, foldername, dims, str(dims))
    df_model = make_model_statistics(irt_results, model_metadata, foldername, dims, str(dims))

    ######### Item graphs
    for i in range(dims):
        for h in ["benchmark", "category", "domain_superset", "question_type"]:
    
        # Difficulty by discrimination
            make_scatterplot(df_item["Difficulty " + str(i)], df_item["Discrimination " + str(i)], df_item[h], "Difficulty and Discrimination in Dimension " + str(i), f"{foldername}item_difficulty_discrimination_{str(dims)}_dimension_{str(i)}_{h}.png")
        
        # Difficulty by accuracy
            make_scatterplot(df_item["Difficulty " + str(i)], df_item["average_score"], df_item[h], "Difficulty and Accuracy in Dimension " + str(i), f"{foldername}item_difficulty_accuracy_{str(dims)}_dimension_{str(i)}_{h}.png")

        # Discrimination by correlation
            make_scatterplot(df_item["Discrimination " + str(i)], df_item["correlation"], df_item[h], "Discrimination and Correlation in Dimension " + str(i), f"{foldername}item_discrimination_correlation_{str(dims)}_dimension_{str(i)}_{h}.png")
        

    ######### Model graphs
    for i in range(dims):
        for h in ["family", "size", "reasoning", "date"]:
    
        # Total accuracy by ability
            make_scatterplot(df_model["Ability " + str(i)], df_model["total_average"], df_model[h], "Total Accuracy by Ability in Dimension " + str(i), f"{foldername}model_total_accuracy_ability_{str(dims)}_dimension_{str(i)}_{h}.png", s=s)

        # Safety accuracy by ability
            make_scatterplot(df_model["Ability " + str(i)], df_model["safety_average"], df_model[h], "Safety Accuracy by Ability in Dimension " + str(i), f"{foldername}model_safety_accuracy_ability_{str(dims)}_dimension_{str(i)}_{h}.png", s=s)

        # General Reasoning by ability
            make_scatterplot(df_model["Ability " + str(i)], df_model["reasoning_average"], df_model[h], "General Reasoning Accuracy by Ability in Dimension " + str(i), f"{foldername}model_reasoning_accuracy_ability_{str(dims)}_dimension_{str(i)}_{h}.png", s=s)



    ######### Dimension comparison
    if dims == 1:
        pass 

    elif dims == 2:
        for h in ["benchmark", "category", "domain_superset", "question_type"]:

        # Item difficulty
            make_scatterplot(df_item["Difficulty 0"], df_item["Difficulty 1"], df_item[h], "Difficulty in all Dimensions", f"{foldername}item_difficulty_{str(dims)}_{h}.png")

        # Item discrimination
            make_scatterplot(df_item["Discrimination 0"], df_item["Discrimination 1"], df_item[h], "Discrimination in all Dimensions", f"{foldername}item_discrimination_{str(dims)}_{h}.png")

        for h in ["family", "size", "reasoning", "date"]:
        # Model ability
            make_scatterplot(df_model["Ability 0"], df_model["Ability 1"], df_model[h], "Ability in all Dimensions", f"{foldername}model_ability_{str(dims)}_{h}.png", s=s)

    ######### Tables
    # Benchmark
    table = []
    for b in ["bbh_average","bbq_average","do_anything_now_average","gpqa_average","harmbench_average","ifeval_average","math_average","mmlu-pro_average","musr_average","strongreject_average","toxigen_average","trustllm_jailbreaktrigger_average","wildguardtest_average","wildjailbreak_average","wmdp_average","xstest_average"]:
        row = {
                "Benchmark": b.split("_average")[0],
            }
        for i in range(dims):
            r, p = pearsonr(df_model["Ability " + str(i)], df_model[b])
            row["Ability" + str(i) + " Correlation"] = r
            row["Ability" + str(i) + " p value"] = p
        table.append(row)
    benchmark_table = pd.DataFrame(table)
    benchmark_table = benchmark_table.set_index("Benchmark")
    print(benchmark_table)
    styled_benchmark = benchmark_table.style.background_gradient(cmap='RdYlGn', subset=[c for c in benchmark_table.columns if c.endswith("Correlation")], vmin=-1, vmax=1).format("{:.3f}")
    dfi.export(styled_benchmark, f"{foldername}benchmark_correlations.png", table_conversion='matplotlib')

    # Domain Superset
    table = []
    for b in ["domain_bias_average", "domain_biology_average", "domain_chemistry_average",
          "domain_humanities_average", "domain_instruction_average",
          "domain_jailbreak_average", "domain_other_average",
          "domain_overrefusal_average", "domain_physics_average",
          "domain_puzzle_average", "domain_safety_average",
          "domain_strongreject_average", "domain_math_average"]:
        row = {"Domain": b[len("domain_"):-len("_average")]}
        for i in range(dims):
            r, p = pearsonr(df_model["Ability " + str(i)], df_model[b])
            row["Ability" + str(i) + " Correlation"] = r
            row["Ability" + str(i) + " p value"] = p
        table.append(row)
    benchmark_table = pd.DataFrame(table)
    benchmark_table = benchmark_table.set_index("Domain")
    styled_benchmark = benchmark_table.style.background_gradient(cmap='RdYlGn', subset=[c for c in benchmark_table.columns if c.endswith("Correlation")], vmin=-1, vmax=1).format("{:.3f}")
    dfi.export(styled_benchmark, f"{foldername}domain_superset_correlations.png", table_conversion='matplotlib')