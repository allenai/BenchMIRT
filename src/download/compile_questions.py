"""
This file collects the safety and capability results into a single file and 
computes the question metadata
"""
import ast
import fire
import json
import string 
import random

import numpy as np
import pandas as pd
import regex as re

from scipy.stats import pointbiserialr

# From https://github.com/huggingface/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/gpqa/utils.py
def preprocess(text):
    """
    Helper function for gpqa formatting
    Input:
        - text: The GPQA prompt text
    Output: 
        - A formatted version of the prompt text
    """
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def main(capability_file="capability_eval_results_filtered_math_verify.csv", safety_file="safety_eval_results.csv"):
    """
    Process the capability and safety data and create a metadata file with the results of both.

    Inputs: 
        capability_file | str: File with collected model results to the general reasoning benchmarks
        safety_file | str: File with collected model results to the safety benchmarks
    Outputs: Saves the results to question_metadata.csv
    """

    capability_eval = pd.read_csv(capability_file, index_col = "question_id")
    categories = []
    average_cols = [c for c in capability_eval.columns if c.startswith("acc_norm")]
    
    # For each capability item, process it according to the benchmark and log metadata
    for i, row in capability_eval.iterrows():
        cat_1 = "" # Internal to benchmark, the most important subset split
        cat_2 = "" # If a benchmark has multiple types of splits, the second type of split
        cat_3 = "" # If a benchmark has multiple types of splits, the third type of split
        bench = "" # The benchmark name
        q = "" # The question/prompt, fully formatted
        av = "" # The average score on the question over all models 
        std = "" # The standard deviation of the score on the question over all models 
        question_type = "" # The format of the question, external to the benchmark
        domain_superset = "" # The general domain of the question, external to the benchmark
        
        if i.startswith("bbh"):
            cat_1 = row["subset"][4:]
            bench = "bbh"
            q = "Q: " + ast.literal_eval(row["question"])["input"] + "\n\nA:"
            av = float(capability_eval.loc[i, average_cols].mean())
            std = capability_eval.loc[i, average_cols].std()
            question_type = "multiple choice logprobs"
            domain_superset = "puzzle"
        
        elif i.startswith("gpqa"):
            cat_1 = ast.literal_eval(row["question"])["High-level domain"]
            cat_2 = ast.literal_eval(row["question"])["Subdomain"]
            cat_3 = row["subset"][5:]
            bench = "gpqa"
            choices = [
                preprocess(ast.literal_eval(row["question"])["Incorrect Answer 1"]), 
                preprocess(ast.literal_eval(row["question"])["Incorrect Answer 2"]), 
                preprocess(ast.literal_eval(row["question"])["Incorrect Answer 3"]),
                preprocess(ast.literal_eval(row["question"])["Correct Answer"])
                ]
            random.shuffle(choices)
            q = "What is the correct answer to this question: " + ast.literal_eval(row["question"])["Question"] + f"\nChoices:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\nAnswer: "
            av = float(capability_eval.loc[i, average_cols].mean())
            std = capability_eval.loc[i, average_cols].std()
            question_type = "multiple choice logprobs"
            domain_superset = cat_1.lower()
       
        elif i.startswith("mmlu"):
            cat_1 = ast.literal_eval(row["question"])["category"]
            bench = "mmlu-pro"
            q = ast.literal_eval(row["question"])["question"] + "\n"
            options = ast.literal_eval(row["question"])["options"]
            for j in range(len(options)):
                q += f"{string.ascii_uppercase[j]}. {options[j]}\n"
            q += "Answer:"
            av = float(capability_eval.loc[i, average_cols].mean())
            std = capability_eval.loc[i, average_cols].std()
            question_type = "multiple choice logprobs"
            if cat_1 in ["bio", "health", "psychology"]:
                domain_superset = "biology"
            elif cat_1 in ["math", "economics", "computer science", "engineering"]:
                domain_superset = "math"
            elif cat_1 in ["business", "history", "law", "philosophy"]:
                domain_superset = "humanities"
            else:
                domain_superset = cat_1.lower()
        
        elif i.startswith("ifeval"):
            cat_1 = "x" 
            bench = "ifeval"
            q = ast.literal_eval(row["question"])["prompt"]
            av = float(capability_eval.loc[i, average_cols].mean())
            std = capability_eval.loc[i, average_cols].std()
            question_type = "generation"
            domain_superset = "instruction"
        
        elif i.startswith("musr"):
            cat_1 = row["subset"][5:]
            bench = "musr"
            q = ast.literal_eval(row["question"])["narrative"] + "\n\n" + ast.literal_eval(row["question"])["question"] + "\n\n"
            choices = ast.literal_eval(ast.literal_eval(row["question"])["choices"])
            for j in range(len(choices)):
                q += f"{j+1} - {choices[j]}\n"
            q += "\nAnswer:"
            av = float(capability_eval.loc[i, average_cols].mean())
            std = capability_eval.loc[i, average_cols].std()
            question_type = "multiple choice logprobs"
            domain_superset = "puzzle"
       
        elif i.startswith("math"):
            cat_1 = ast.literal_eval(row["question"])["type"]
            bench = "math"
            q = "Problem:" + "\n" + ast.literal_eval(row["question"])["problem"] + "\n\n" + "Solution:"
            av = float(capability_eval.loc[i, average_cols].mean())
            std = capability_eval.loc[i, average_cols].std()
            question_type = "generation"
            domain_superset = "math"

        d = {
            "question_id": i,
            "benchmark": bench,
            "category": "General Reasoning",
            "subcategory_one": cat_1,
            "subcategory_two": cat_2,
            "subcategory_three": cat_3,
            "question_text": q,
            "average_score": av,
            "standard_deviation": std,
            "question_type": question_type,
            "domain_superset": domain_superset
        }
        categories.append(d)

    # Process the safety eval results
    task_types = [
        {
            "prompt": "instruction",
            "subset": None,
            "correct": "response_refusal",
        },
        {
            "prompt": "prompt",
            "subset": "subcategory",
            "correct": "response_harmfulness",
        },
        {
            "prompt": "Behavior",
            "subset": "FunctionalCategory",
            "correct": "response_harmfulness",
        },
        {
            "prompt": "input",
            "subset": None,
            "correct": "classification",
        },
        {
            "prompt": "prompt",
            "subset": "type",
            "correct": "xstest",
        },
        {
            "prompt": "instruction",
            "subset": None,
            "correct": "benign",
        },
        {
            "prompt": "instruction",
            "subset": None,
            "correct": "response_refusal",
        },
        {
            "prompt": "instruction",
            "subset": "label",
            "correct": "response_refusal",
        },
        {
            "prompt": "formatted_question",
            "subset": "category",
            "correct": "analytics",
        },
        {
            "prompt": "formatted_question",
            "subset": "category",
            "correct": "answer_choice",
        },
        {
            "prompt": "jailbroken_prompt",
            "subset": "category",
            "correct": "response_score",
        },
        ]
    tasks = [
                "wildjailbreak:harmful",
                "wildguardtest:default",
                "harmbench:default",
                "toxigen:tiny",
                "xstest:default",
                "wildjailbreak:benign",
                "do_anything_now:default",
                "trustllm_jailbreaktrigger:default",
                "bbq:default",
                "wmdp:default",
                "strongreject:logprobs",
            ]
    # Identify subset by id number
    toxigen_ids = {
        "0": "asian",
        "1": "black",
        "2": "chinese",
        "3": "jewish",
        "4": "latino",
        "5": "lgbtq",
        "6": "mental_disability",
        "7": "mexican",
        "8": "middle_east",
        "9": "muslim",
        "10": "native_american",
        "11": "physical_diability",
        "12": "trans",
        "13": "women"
    }

    safety_averages = pd.read_csv(safety_file, index_col = "question_id")
    s_average_cols = [c for c in safety_averages.columns if c.startswith("acc_norm")]

    for task, task_type in zip(tasks, task_types):
        file_task = task.replace(":", "_")
        with open(f"./safety_output/01-ai__Yi-34B-Chat/results.{file_task}", "r") as f:
            data = json.load(f)[task]
        for prompt in data:
            if not str(prompt["id"]).startswith("Best"):
                if task_type["subset"] == None:
                    subset = task_type["correct"]
                else:
                    try:
                        subset = prompt[task_type["subset"]]
                    except:
                        print(prompt)
                
                question_id = task.split(":")[0] + "__" + str(subset) + "_" + str(str(prompt["id"]).split("/")[-1])
                cat_1 = "" # Internal to benchmark, the most important subset split
                cat_2 = "" # If a benchmark has multiple types of splits, the second type of split
                cat_3 = "" # If a benchmark has multiple types of splits, the third type of split
                bench = "" # The benchmark name
                q = "" # The question/prompt, fully formatted
                av = "" # The average score on the question over all models 
                std = "" # The standard deviation of the score on the question over all models 
                question_type = "" # The format of the question, external to the benchmark
                domain_superset = "" # The general domain of the question, external to the benchmark
                domain_superset = ""
                
                if task.startswith("wildjailbreak:harmful"):
                    cat_1 = "harmful"
                    bench = "wildjailbreak"
                    q = prompt["instruction"]
                    question_type = "generation - safety"
                    domain_superset = "jailbreak"
                
                elif task.startswith("wildjailbreak:benign"):
                    cat_1 = "benign"
                    bench = "wildjailbreak" 
                    q = prompt["instruction"]
                    question_type = "generation - benign"
                    domain_superset = "overrefusal"
                
                elif task.startswith("wildguardtest"):
                    cat_1 = prompt["subcategory"]
                    cat_2 = prompt["prompt_type"]
                    bench = "wildguardtest"
                    q = prompt["prompt"]
                    question_type = "generation - safety"
                    if cat_2 == "vanilla":
                        domain_superset = "safety"
                    elif cat_2 == "adversarial":
                        domain_superset = "jailbreak"
                
                elif task.startswith("harmbench"):
                    cat_1 = prompt["FunctionalCategory"]
                    cat_2 = prompt["SemanticCategory"]
                    bench = "harmbench"
                    if cat_1 == "contextual":
                        q = prompt["ContextString"] + "\n\n---\n\n" + prompt["Behavior"]
                    else:
                        q = prompt["Behavior"]
                    question_type = "generation - safety"
                    domain_superset = "safety"
                
                elif task.startswith("xstest"):
                    cat_1 = prompt["type"]
                    bench = "xstest"
                    q = prompt["prompt"]
                    if cat_1.startswith("contrast"):
                        question_type = "generation - safety"
                        domain_superset = "safety"
                    else:
                        question_type = "generation - benign"
                        domain_superset = "overrefusal"
                
                elif task.startswith("do_anything_now"):
                    cat_1 = prompt["source"]
                    bench = "do_anything_now"
                    q = prompt["instruction"]
                    question_type = "generation - safety"
                    domain_superset = "jailbreak"
                
                elif task.startswith("trustllm"):
                    cat_1 = prompt["label"][0]
                    cat_2 = prompt["source"]
                    bench = "trustllm_jailbreaktrigger"
                    q = prompt["instruction"]
                    question_type = "generation - safety"
                    domain_superset = "jailbreak"
                
                elif task.startswith("bbq"):
                    cat_2 = prompt["question_polarity"]
                    cat_1 = prompt["context_condition"]
                    cat_3 = prompt["category"]
                    bench = "bbq"
                    q = prompt["formatted_question"]
                    question_type = "multiple choice generation"
                    if cat_1 == "ambig":
                        domain_superset = "bias"
                    elif cat_1 == "disambig":
                        domain_superset = "puzzle"
                
                elif task.startswith("wmdp"):
                    cat_1 = prompt["category"]
                    bench = "wmdp"
                    q = prompt["formatted_question"]
                    question_type = "multiple choice generation"
                    if cat_1 == "wmdp-bio":
                        domain_superset = "biology"
                    elif cat_1 == "wmdp-chem":
                        domain_superset = "chemistry"
                    elif cat_1 == "wmdp-cyber":
                        domain_superset = "math"
                
                elif task.startswith("strongreject"):
                    cat_1 = prompt["category"]
                    cat_2 = prompt["forbidden_prompt"]
                    cat_3 = prompt["jailbreak"]
                    bench = "strongreject"
                    q = prompt["jailbroken_prompt"]
                    question_type = "generation - safety"
                    domain_superset = "strongreject"
                
                elif task.startswith("toxigen"):
                    id_range = str(int(prompt["id"].split("/")[-1]) // 100)
                    cat_1 = toxigen_ids[id_range]
                    bench = "toxigen"
                    q = prompt["input"]
                    question_type = "generation - safety"
                    domain_superset = "bias"
                
                d = {
                "question_id": question_id,
                "benchmark": bench,
                "category": "Safety",
                "subcategory_one": cat_1,
                "subcategory_two": cat_2,
                "subcategory_three": cat_3,
                "question_text": q,
                "average_score": safety_averages.loc[question_id, s_average_cols].mean(),
                "standard_deviation": safety_averages.loc[question_id, s_average_cols].std(),
                "question_type": question_type,
                "domain_superset": domain_superset
                }
                categories.append(d)
    
    df = pd.DataFrame(categories)
    df = df.set_index("question_id")

    # Calculate averages for point-biserial correlation to benchmark calculation
    c = capability_eval[average_cols].join(df["benchmark"], how="left")
    s = safety_averages[s_average_cols].join(df["benchmark"], how="left")

    c_averages = c.groupby("benchmark").mean()
    s_averages = s.groupby("benchmark").mean()

    # Calculate the correlation for each item to its benchmark
    for i, row in df.iterrows():
        if row["category"] == "Safety":
            corr = pointbiserialr(s_averages.loc[row["benchmark"]].to_numpy(dtype=float), safety_averages.loc[i, s_average_cols].to_numpy(dtype=float))
            # print(corr)
        else:
            corr = pointbiserialr(c_averages.loc[row["benchmark"]].to_numpy(dtype=float), capability_eval.loc[i, average_cols].to_numpy(dtype=float))
            # print(corr)
        if np.isnan(corr[0]):
            df.loc[i, "correlation"] = 0
        else:
            df.loc[i, "correlation"] = corr[0]

    # Calculate averages for point-biserial correlation to subset calculation
    c = capability_eval[average_cols].join(df[["benchmark", "subcategory_one"]], how="left")
    s = safety_averages[s_average_cols].join(df[["benchmark", "subcategory_one"]], how="left")

    # Calculate the correlation for each item to its first subset
    c_averages = c.groupby(by=["benchmark", "subcategory_one"]).mean()
    s_averages = s.groupby(by=["benchmark", "subcategory_one"]).mean()

    for i, row in df.iterrows():
        if row["category"] == "Safety":
            corr = pointbiserialr(s_averages.loc[(row["benchmark"], row["subcategory_one"])].to_numpy(dtype=float), safety_averages.loc[i, s_average_cols].to_numpy(dtype=float))
            # print(corr)
        else:
            corr = pointbiserialr(c_averages.loc[(row["benchmark"], row["subcategory_one"])].to_numpy(dtype=float), capability_eval.loc[i, average_cols].to_numpy(dtype=float))
            # print(corr)
        if np.isnan(corr[0]):
            df.loc[i, "correlation_subset"] = 0
        else:
            df.loc[i, "correlation_subset"] = corr[0]


    df.to_csv("question_metadata.csv")
