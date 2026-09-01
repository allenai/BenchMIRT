"""
This file computes correlations between benchmarks and dimensions, 
as well as between dimensions of different runs
"""
all_benchmarks = {
    "bbh": [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies"
    ],
    "bbq": [
        "ambig",
        "disambig"],
    "do_anything_now": [
        "jailbreak_chat",
        "LLM Promptwriting",
        "BreakGPT",
        "ChatGPTJailbreak",
        "ChatGPT",
        "AI Prompt Sharing"],
    "gpqa": [
        "Biology",
        "Physics",
        "Chemistry"],
    "harmbench": [
        "standard",
        "copyright",
        "contextual"],
    "ifeval": ["x"],
    "math": [
        "Algebra",
        "Counting & Probability",
        "Geometry",
        "Intermediate Algebra",
        "Number Theory",
        "Prealgebra",
        "Precalculus"],
    "mmlu-pro": [
        "business",
        "law",
        "physics",
        "computer science",
        "philosophy",
        "engineering",
        "psychology",
        "biology",
        "chemistry",
        "history",
        "other",
        "health",
        "economics",
        "math"],
    "musr": [
        "murder_mysteries",
        "object_placements",
        "team_allocation"],
    "strongreject": [
        "Hate, harassment and discrimination",
        "Disinformation and deception",
        "Violence",
        "Illegal goods and services",
        "Non-violent crimes",
        "Sexual content"],
    "toxigen": [
        "asian",
        "black",
        "chinese",
        "jewish",
        "latino",
        "lgbtq",
        "mental_disability",
        "mexican",
        "middle_east",
        "muslim",
        "native_american",
        "physical_diability",
        "trans",
        "women"
    ],
    "trustllm_jailbreaktrigger": [
        "bad words",
        "programming",
        "fixed sentence",
        "cot",
        "Refusal sentence prohibition",
        "json_format",
        "no_punctuation",
        "leetspeak",
        "no_long_word",
        "without_the",
        "url_encode",
        "cou",
        "scenario",
        "multitask"],
    "wildguardtest": [
        "private_information_individual",
        "others",
        "cyberattack",
        "sexual_content",
        "social_stereotypes_and_unfair_discrimination",
        "toxic_language_hate_speech",
        "violence_and_physical_harm",
        "defamation_encouraging_unethical_or_unsafe_actions",
        "fraud_assisting_illegal_activities",
        "mental_health_over-reliance_crisis",
        "causing_material_harm_by_disseminating_misinformation",
        "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns",
        "copyright_violations",
        "sensitive_information_organization_government"],
    "wildjailbreak": [
        "harmful",
        "benign"],
    "wmdp": [
        "wmdp-bio",
        "wmdp-chem",
        "wmdp-cyber"],
    "xstest": [
        "homonyms",
        "contrast_homonyms",
        "figurative_language",
        "contrast_figurative_language",
        "safe_targets",
        "contrast_safe_targets",
        "safe_contexts",
        "contrast_safe_contexts",
        "definitions",
        "contrast_definitions",
        "nons_group_real_discr",
        "real_group_nons_discr",
        "contrast_discr",
        "historical_events",
        "contrast_historical_events",
        "privacy_public",
        "privacy_fictional",
        "contrast_privacy"],
}

domain_set = ["bias","biology","chemistry","humanities","instruction","jailbreak","math","other","overrefusal","physics","puzzle","strongreject","safety"]
from scipy.stats import pearsonr
import pandas as pd

def main(one_dim, two_dim, three_dim, four_dim, output_dir, cat = "benchmark"):
    """
    Read in the IRT parameters from all four dimensional runs and calculate their correlations
    Input:
        one_dim | str: The filename of the IRT parameter output of a one-dimensional IRT run
        two_dim | str: The filename of the MIRT parameter output of a two-dimensional MIRT run
        three_dim | str: The filename of the MIRT parameter output of a three-dimensional MIRT run
        four_dim | str: The filename of the MIRT parameter output of a four-dimensional MIRT run
        output_dir | str: The foldername in which to save the results
        cat | str: The granularity at which to run the correlations (benchmark, domain, subcat)
    Output:
        Saves two csvs, one with the cat benchmark correlations to each dimension from the IRT models,
        one with the correlations between each dimension
    """
    # Pick the granularity to correlate at before reading anything off disk    
    if cat == "benchmark":
        categories, prefix = list(all_benchmarks.keys()), ""
    elif cat == "domain":
        categories, prefix = domain_set, "domain_"
    elif cat == "subcat":
        categories, prefix = [s for k in all_benchmarks for s in all_benchmarks[k]], "subcat_"
   
    one_dim_model = pd.read_csv(one_dim)
    two_dim_model = pd.read_csv(two_dim)
    three_dim_model = pd.read_csv(three_dim)
    four_dim_model = pd.read_csv(four_dim)

    # pearsonr pairs values by position, so the four runs have to be row-aligned by
    # model. Index each run by model name, check they cover the same models, then put
    # the multidimensional runs in the same order as the one dimensional run.
    runs = {
        "one_dim": one_dim_model,
        "two_dim": two_dim_model,
        "three_dim": three_dim_model,
        "four_dim": four_dim_model
    }
    for name, model_df in runs.items():
        if "model" not in model_df.columns:
            raise ValueError(f"{name} file has no 'model' column, cannot align the runs by model")
        model_df.set_index("model", inplace=True)
        if model_df.index.has_duplicates:
            duplicates = sorted(model_df.index[model_df.index.duplicated()].unique())
            raise ValueError(f"{name} file lists the same model more than once: {duplicates}")

    reference = one_dim_model.index
    for name, model_df in runs.items():
        if set(model_df.index) != set(reference):
            raise ValueError(
                f"{name} covers different models than one_dim "
                f"(missing {sorted(set(reference) - set(model_df.index))}, "
                f"unexpected {sorted(set(model_df.index) - set(reference))})"
            )

    two_dim_model = two_dim_model.reindex(reference)
    three_dim_model = three_dim_model.reindex(reference)
    four_dim_model = four_dim_model.reindex(reference)

    averages = {}


    for b in categories:
        one_dim_b = one_dim_model[prefix + b+"_average"]
        two_dim_b = two_dim_model[prefix + b+"_average"]
        three_dim_b = three_dim_model[prefix + b+"_average"]
        four_dim_b = four_dim_model[prefix + b+"_average"]

        if not one_dim_b.equals(two_dim_b):
            print("One to Two, ", b)
        if not three_dim_b.equals(two_dim_b):
            print("Two to Three, ", b)
        if not three_dim_b.equals(four_dim_b):
            print("Three to Four, ", b)
        
        averages[b] = one_dim_b
    
    
    averages["1_d_1_a"] = one_dim_model["Ability 0"]

    averages["2_d_1_a"] = two_dim_model["Ability 0"]
    averages["2_d_2_a"] = two_dim_model["Ability 1"]

    averages["3_d_1_a"] = three_dim_model["Ability 0"]
    averages["3_d_2_a"] = three_dim_model["Ability 1"]
    averages["3_d_3_a"] = three_dim_model["Ability 2"]

    averages["4_d_1_a"] = four_dim_model["Ability 0"]
    averages["4_d_2_a"] = four_dim_model["Ability 1"]
    averages["4_d_3_a"] = four_dim_model["Ability 2"]
    averages["4_d_4_a"] = four_dim_model["Ability 3"]

    benchmarks_csv = []
    for b in categories:
        r = {}
        r["benchmark"] = b
        r["1_d_1_a"], r["1_d_1_a p"] = pearsonr(averages[b], averages["1_d_1_a"])
        r["2_d_1_a"], r["2_d_1_a p"] = pearsonr(averages[b], averages["2_d_1_a"])
        r["2_d_2_a"], r["2_d_2_a p"] = pearsonr(averages[b], averages["2_d_2_a"])
        r["3_d_1_a"], r["3_d_1_a p"] = pearsonr(averages[b], averages["3_d_1_a"])
        r["3_d_2_a"], r["3_d_2_a p"] = pearsonr(averages[b], averages["3_d_2_a"])
        r["3_d_3_a"], r["3_d_3_a p"] = pearsonr(averages[b], averages["3_d_3_a"])
        r["4_d_1_a"], r["4_d_1_a p"] = pearsonr(averages[b], averages["4_d_1_a"])
        r["4_d_2_a"], r["4_d_2_a p"] = pearsonr(averages[b], averages["4_d_2_a"])
        r["4_d_3_a"], r["4_d_3_a p"] = pearsonr(averages[b], averages["4_d_3_a"])
        r["4_d_4_a"], r["4_d_4_a p"] = pearsonr(averages[b], averages["4_d_4_a"])
    
        benchmarks_csv.append(r)
    benchmarks_c = pd.DataFrame(benchmarks_csv)
    
    benchmarks_c.to_csv(f"{output_dir}{cat}_corr.csv")

    abilities = ["1_d_1_a", "2_d_1_a", "2_d_2_a", "3_d_1_a", "3_d_2_a", "3_d_3_a","4_d_1_a", "4_d_2_a", "4_d_3_a", "4_d_4_a" ]
    abilities_csv = []
    for i in range(len(abilities)):
        r = {"primary": abilities[i]}
        for j in range(i, len(abilities)):
            r[abilities[j]], r[abilities[j] + " p"]  = pearsonr(averages[abilities[i]], averages[abilities[j]])
        abilities_csv.append(r)
    abilities_c = pd.DataFrame(abilities_csv)
    abilities_c.to_csv(f"{output_dir}abilities_corr_{cat}.csv")

