"""
This file downloads and formats the general reasoning benchmark results
It automatically uses the models listed in models_static.py
download() requires manual continuance for every model downloaded from huggingface

The correct run order is:
download()
download_math_to_csv()
process_math_verify()
replace_math_scores_with_verified()


"""
import ast
import json

import pandas as pd 

from datasets import load_dataset
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from models_static import non_reasoning_models, reasoning_models


# Define models to be run, and the eval_subset pairs to download
models = non_reasoning_models + reasoning_models 
evals = ["bbh", "gpqa", "math", "mmlu_pro", "musr", "ifeval"]
correctness = ["acc_norm", "acc_norm", "exact_match", "acc", "acc_norm", "prompt_level_strict_acc"]
subsets = [
    [
        "_boolean_expressions"
        , "_causal_judgement"
        , "_date_understanding"
        , "_disambiguation_qa"
        , "_formal_fallacies"
        , "_geometric_shapes"
        , "_hyperbaton"
        , "_logical_deduction_five_objects"
        , "_logical_deduction_seven_objects"
        , "_logical_deduction_three_objects"
        , "_movie_recommendation"
        , "_navigate"
        , "_object_counting"
        , "_penguins_in_a_table"
        , "_reasoning_about_colored_objects"
        , "_ruin_names"
        , "_salient_translation_error_detection"
        , "_snarks"
        , "_sports_understanding"
        , "_temporal_sequences"
        , "_web_of_lies"
        , "_tracking_shuffled_objects_three_objects"
        , "_tracking_shuffled_objects_five_objects"
        , "_tracking_shuffled_objects_seven_objects"
    ],
    [
        "_extended"
    ],
    [
        "_algebra_hard"
        , "_counting_and_prob_hard"
        , "_geometry_hard"
        , "_intermediate_algebra_hard"
        , "_num_theory_hard"
        , "_prealgebra_hard"
        , "_precalculus_hard"
    ],
    [""],
    [
        "_murder_mysteries"
        , "_object_placements"
        , "_team_allocation"
    ], 
    [""]

]

# Modified slightly from the original MathVerify.process_answers() to utiliize Latex and Expr parsing
def math_verify_process_answers(df: pd.DataFrame, gold_is_latex: bool) -> pd.DataFrame:
    """Process each answer through the sympy extraction workflow and compare with gold using math_verify."""
    results = []
    
    correct_count = 0
    total_count = 0
    
    # Create the verification function
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),ExprExtractionConfig()), # Modified to include both
        pred_extraction_target=(LatexExtractionConfig(),ExprExtractionConfig()), # Modified to include both
        aggregation_function=max,
        precision=6
    )
    
    for _, row in df.iterrows():
        extracted_answers = None
        gold_answers = None
        grade = 0
        try:
            # Use the verification function
            grade, extracted_answers = verify_func([row['gold']], [row['answer']])
            
            if extracted_answers is None:
                extracted_answers = None
                gold_answers = None
            else:
                gold_answers = extracted_answers[0]
                extracted_answers = extracted_answers[1]

            total_count += 1
            if grade == 1:
                correct_count += 1
            
            result = {
                'question_id': row['question_id'], # Modified to add question_id
                'subset': row['subset'],
                'original_answer': row['answer'],
                'gold_answer': row['gold'],
                'extracted_answer': extracted_answers,
                'extracted_gold': gold_answers,
                'is_correct': grade == 1
            }
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'question_id': row['question_id'], # Modified to add question_id
                'subset': row['subset'],
                'original_answer': row['answer'],
                'gold_answer': row['gold'],
                'extracted_answer': extracted_answers,
                'extracted_gold': gold_answers,
                'is_correct': grade == 1,
                'error': str(e)
            })
    
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\nEvaluation Results:")
    print(f"Total examples: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Add summary stats to the dataframe
    results_df.attrs['accuracy'] = accuracy
    results_df.attrs['total_count'] = total_count
    results_df.attrs['correct_count'] = correct_count
    
    return results_df

def download():
    """
    Code to download and process the first pass of the capability evals.
    This does not correct the math scores.
    This function requires an HF_TOKEN that has been granted access to the OpenLLM Leaderboard dataset

    Inputs: None, but uses globally defined models, evals, correctness, subsets
    Outputs: Saves the results to capability_eval_results.csv
    """
    # Load the file if it exists, create it if not
    try:
        big_df = pd.read_csv("capability_eval_results.csv", index_col = "question_id")
        big_df['question'] = big_df['question'].apply(ast.literal_eval)
    except:
        print("Failed to find existing results, creating a new file")
        big_df = pd.DataFrame({"question_id": []})
        big_df = big_df.set_index("question_id")
    
    # Loop through the globally defined models
    for i in range(len(models)):
        # Allows for controlling the rate at which you request data
        input("Press Enter to continue...")
        model_csv = []

        # Stack each eval__subset pair
        for j in range(len(evals)):
            for k in range(len(subsets[j])):
                dataset_name = "open-llm-leaderboard/" + models[i] + "-details"
                subset_name = f"{models[i]}__leaderboard_{evals[j]}{subsets[j][k]}"
                print(dataset_name)
                print(subset_name)
                data = load_dataset(
                    dataset_name,
                    name=subset_name,
                    split="latest"
                )
                data = pd.DataFrame(data)
                for _, row in data.iterrows():
                    if row[correctness[j]] in [1, 1.0, '1', '1.0', True, 'True']:
                        acc_norm = 1
                    elif row[correctness[j]] in [0, 0.0, '0', '0.0', False, 'False']:
                        acc_norm = 0
                    else:
                        print("acc_norm is ", row[correctness[j]])
                        return
                    question_id = evals[j] + "_" + subsets[j][k] + "_" + str(row["doc_id"])
                    if "question" in big_df.columns:
                        # These mmlu questions were edited during the time the models were run, so we exclude them
                        if question_id not in [
                                                "mmlu_pro__72", "mmlu_pro__6228", "mmlu_pro__2528", "mmlu_pro__7832", "mmlu_pro__249", "mmlu_pro__8385",
                                                "mmlu_pro__8232", "mmlu_pro__8360", "mmlu_pro__8592", "mmlu_pro__8696", "mmlu_pro__8744", "mmlu_pro__5313",
                                                "mmlu_pro__8569", "mmlu_pro__7938", "mmlu_pro__8506", "mmlu_pro__8051", "mmlu_pro__8395", "mmlu_pro__8443",
                                                "mmlu_pro__11067", "mmlu_pro__8260", "mmlu_pro__1205", "mmlu_pro__2293", "mmlu_pro__2701", "mmlu_pro__7597",
                                                "mmlu_pro__8045", "mmlu_pro__8397", "mmlu_pro__8493", "mmlu_pro__8693", "mmlu_pro__6894", "mmlu_pro__7926",
                                                "mmlu_pro__7958", "mmlu_pro__8078", "mmlu_pro__8574", "mmlu_pro__8694", "mmlu_pro__8766", "mmlu_pro__8223", 
                                                "mmlu_pro__8431", "mmlu_pro__5887"

                                                ]: # Known discrepancies
                            
                            # Check the answers and the questions -- allow for the order of the answers to change
                            if evals[j] == "gpqa": # Answer choices may switch order
                                if big_df.loc[question_id, "question"]["Question"] == row["doc"]["Question"] \
                                    and big_df.loc[question_id, "question"]["Correct Answer"] == row["doc"]["Correct Answer"] \
                                    and big_df.loc[question_id, "question"]["Incorrect Answer 1"] == row["doc"]["Incorrect Answer 1"] \
                                    and big_df.loc[question_id, "question"]["Incorrect Answer 2"] == row["doc"]["Incorrect Answer 2"] \
                                    and big_df.loc[question_id, "question"]["Incorrect Answer 3"] == row["doc"]["Incorrect Answer 3"]:
                                    pass 
                                else:
                                    print(f"{question_id}\n{big_df.loc[question_id, "question"]} \n\n {row["doc"]}")
                            
                            # Check just the questions
                            elif evals[j] == "math": # Formatting on the answer choices changes
                                if big_df.loc[question_id, "question"]["problem"] == row["doc"]["problem"]:
                                    pass
                                else:
                                    print(f"{question_id}\n{big_df.loc[question_id, "question"]} \n\n {row["doc"]}")
                            
                            # Verify the question and metadt match if it is not a known exception
                            else:
                                if big_df.loc[question_id, "question"] != row["doc"]:
                                    print(type(big_df.loc[question_id, "question"]), type(row["doc"]))
                                    print(f"{question_id}\n{big_df.loc[question_id, "question"]} \n\n {row["doc"]}")
                                if big_df.loc[question_id, "subset"] != evals[j] + subsets[j][k]:
                                    print(f"{question_id}\n{big_df.loc[question_id, "subset"]} \n\n {evals[j] + subsets[j][k]}")

                            r = {
                                "question_id": question_id,
                                f"acc_norm_{models[i]}": acc_norm
                            }
                            model_csv.append(r)
                    
                    # For the first model, load everything that is not a known discrepancy and include a question and subset column
                    else:
                        if question_id not in [
                                                "mmlu_pro__72", "mmlu_pro__6228", "mmlu_pro__2528", "mmlu_pro__7832", "mmlu_pro__249", "mmlu_pro__8385",
                                                "mmlu_pro__8232", "mmlu_pro__8360", "mmlu_pro__8592", "mmlu_pro__8696", "mmlu_pro__8744", "mmlu_pro__5313",
                                                "mmlu_pro__8569", "mmlu_pro__7938", "mmlu_pro__8506", "mmlu_pro__8051", "mmlu_pro__8395", "mmlu_pro__8443",
                                                "mmlu_pro__11067", "mmlu_pro__8260", "mmlu_pro__1205", "mmlu_pro__2293", "mmlu_pro__2701", "mmlu_pro__7597",
                                                "mmlu_pro__8045", "mmlu_pro__8397", "mmlu_pro__8493", "mmlu_pro__8693", "mmlu_pro__6894", "mmlu_pro__7926",
                                                "mmlu_pro__7958", "mmlu_pro__8078", "mmlu_pro__8574", "mmlu_pro__8694", "mmlu_pro__8766", "mmlu_pro__8223", 
                                                "mmlu_pro__8431", "mmlu_pro__5887"

                                                ]: # Known discrepancies
                            r = {
                                "question_id": question_id,
                                f"question": row["doc"],
                                f"subset": evals[j] + subsets[j][k],
                                f"acc_norm_{models[i]}": acc_norm
                            }
                            model_csv.append(r)
        # Save the results
        model_df = pd.DataFrame(model_csv)
        model_df = model_df.set_index("question_id")
        big_df = pd.merge(big_df, model_df, on='question_id', how='outer')
        big_df.to_csv("capability_eval_results.csv")


def download_math_to_csv():
    """
    Download the individual math results and store the exact generated answer 
    to the question for rescoring with MathVerify.

    Inputs: None, but relies on globally defined models
    Outputs: Saves the results to math/math_scores_{model}.csv
    """
    for model in models:
        subsets = [
            "_algebra_hard"
            , "_counting_and_prob_hard"
            , "_geometry_hard"
            , "_intermediate_algebra_hard"
            , "_num_theory_hard"
            , "_prealgebra_hard"
            , "_precalculus_hard"
        ]
        final_csv = []
        for i in range(len(subsets)):
            dataset_name = "open-llm-leaderboard/" + model + "-details"
            subset_name = f"{model}__leaderboard_math{subsets[i]}"
            print(dataset_name)
            print(subset_name)
            data = load_dataset(
                dataset_name,
                name=subset_name,
                split="latest"
            )
            data = pd.DataFrame(data)
            for _, row in data.iterrows():
                new_row = {
                    "question_id": "math_" + subsets[i] + "_" + str(row["doc_id"]),
                    "subset": subsets[i],
                    "answer": row["filtered_resps"][0],
                    "gold": row["target"]
                }
                final_csv.append(new_row)
        df = pd.DataFrame(final_csv)
        df.to_csv(f"math/math_scores_{model}.csv")


def process_math_verify():
    """
    Rescore the individual results with MathVerify and the most recent answers from the Math dataset

    Inputs: None, but relies on globally defined models
    Outputs: Saves the results to math/verified_math_scores_{model}.csv
    """

    # Use the QWQ model answers as gold as they are the most updated
    gold_df = pd.read_csv("math/math_scores_Qwen__QwQ-32B.csv")

    for model in models:
        input_csv = f"math/math_scores_{model}.csv"
        output_csv = f"math/verified_math_scores_{model}.csv"
        input_df = pd.read_csv(input_csv).drop(columns=["gold"])
        # input_df = pd.read_csv(input_csv)
        input_df = input_df.merge(gold_df[["question_id", "gold"]], on="question_id", how="left")
    
        # Process answers and extract sympy objects
        results_df = math_verify_process_answers(input_df, True)
        
        # Save results to output CSV
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
        df = pd.read_csv(output_csv)
        model_scores = df.groupby([f"subset"])[f"is_correct"].mean()
        print(model_scores)

def replace_math_scores_with_verified():
    """
    Replace the original result download Math scores with the updated Math scores

    Inputs: None, but relies on globally defined models
    Outputs: Saves the results to capability_eval_results_filtered_math_verify.csv
    """
    capability = pd.read_csv("capability_eval_results.csv", index_col = "question_id")
    for model in models:
        print(capability.groupby([f"subset"])[f"acc_norm_{model}"].mean())
        new_math = pd.read_csv(f"math/verified_math_scores_{model}.csv")
        for i, row in new_math.iterrows():
            capability.loc[capability.index == row["question_id"], f"acc_norm_{model}"] = int(row["is_correct"])
        print(new_math.groupby([f"subset"])[f"is_correct"].mean())
        print(capability.groupby([f"subset"])[f"acc_norm_{model}"].mean())
    capability.to_csv("capability_eval_results_filtered_math_verify.csv")
