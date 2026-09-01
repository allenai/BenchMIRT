"""
This file supports MIRT item filtering 
"""
import pandas as pd
import copy
from scipy.stats import pearsonr
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from evaluate_constants import all_benchmarks

def main(input_data, num_bins, percent_cut, data_filter, input_model, all_responses, output_dir, dim_0_neg=False, dim_1_neg=False):
    """
    Given a set of results from a two dimensional MIRT run, filter on each dimension to the most discriminative items
    Inputs:
        input_data | str: The filename of the csv containing the MIRT results
        num_bins | int: The number of item-difficulty based bins to split into, set to 100 for the paper
        percent_cut | float: The percentage of data to exclude, set to 0.5 or 0.9 for the paper
        data_filter | str: Filter all items together, by benchmark, or by category
        input_model | str: The filepath to the csv with metadata on each model
        all_responses | str: The filepath to a csv with original responses from each LLM
        output_dir | str: The folder in which to save the results
        dim_0_neg | bool: whether the dimension 0 ability is negative, and thus the most discriminative items are negative
        dim_1_neg | bool: whether the dimension 1 ability is negative, and thus the most discriminative items are negative
    Output:
        Saves eight csvs:
            - A csv with all of the data, and the addition of the difficulty bin, discrimination, dim0_included and dim1_included columns
            - A csv with all of the original response data for the items included by dimension 0
            - A csv with all of the original response data for the items included by dimension 1
            - A csv with benchmark averages on the items filtered by dimension 0
            - A csv with benchmark averages on the items filtered by dimension 1
            - A csv adding ability scores to items filtered by dimension 0
            - A csv adding ability scores to items filtered by dimension 1
            - The final csv, with correlations between abilities, original averages, and filtered averages

    """

    df = pd.read_csv(input_data)
    read_df = copy.deepcopy(df).set_index("question_id")
    if data_filter == "category":
        final_df = pd.DataFrame()
        for cat in df["category"].unique():
            cat_df = copy.deepcopy(df[df["category"] == cat])
            cat_df['diff_0_bin'] = pd.qcut(cat_df['Difficulty 0'], num_bins, labels=False)
            cat_df['diff_1_bin'] = pd.qcut(cat_df['Difficulty 1'], num_bins, labels=False)
            cat_df['disc_0_abs'] = cat_df['Discrimination 0']
            cat_df['disc_1_abs'] = cat_df['Discrimination 1']
            
            if dim_0_neg != False:
                dim0_threshold = cat_df.groupby('diff_0_bin')['disc_0_abs'].transform(lambda x: x.quantile(1 - percent_cut))
                cat_df['dim0_included'] = cat_df['disc_0_abs'] <= dim0_threshold
            else:
                dim0_threshold = cat_df.groupby('diff_0_bin')['disc_0_abs'].transform(lambda x: x.quantile(percent_cut))
                cat_df['dim0_included'] = cat_df['disc_0_abs'] >= dim0_threshold
            if dim_1_neg != False:
                dim1_threshold = cat_df.groupby('diff_1_bin')['disc_1_abs'].transform(lambda x: x.quantile(1 - percent_cut))
                cat_df['dim1_included'] = cat_df['disc_1_abs'] <= dim1_threshold
            else:
                dim1_threshold = cat_df.groupby('diff_1_bin')['disc_1_abs'].transform(lambda x: x.quantile(percent_cut))
                cat_df['dim1_included'] = cat_df['disc_1_abs'] >= dim1_threshold
            final_df = pd.concat([final_df, cat_df], axis=0, ignore_index=True)
    elif data_filter == "benchmark":
        final_df = pd.DataFrame()
        for cat in df["benchmark"].unique():
            cat_df = copy.deepcopy(df[df["benchmark"] == cat])
            cat_df['diff_0_bin'] = pd.qcut(cat_df['Difficulty 0'], num_bins, labels=False)
            cat_df['diff_1_bin'] = pd.qcut(cat_df['Difficulty 1'], num_bins, labels=False)
            cat_df['disc_0_abs'] = cat_df['Discrimination 0']
            cat_df['disc_1_abs'] = cat_df['Discrimination 1']
            if dim_0_neg != False:
                dim0_threshold = cat_df.groupby('diff_0_bin')['disc_0_abs'].transform(lambda x: x.quantile(1 - percent_cut))
                cat_df['dim0_included'] = cat_df['disc_0_abs'] <= dim0_threshold
            else:
                dim0_threshold = cat_df.groupby('diff_0_bin')['disc_0_abs'].transform(lambda x: x.quantile(percent_cut))
                cat_df['dim0_included'] = cat_df['disc_0_abs'] >= dim0_threshold
            if dim_1_neg != False:
                dim1_threshold = cat_df.groupby('diff_1_bin')['disc_1_abs'].transform(lambda x: x.quantile(1 - percent_cut))
                cat_df['dim1_included'] = cat_df['disc_1_abs'] <= dim1_threshold
            else:
                dim1_threshold = cat_df.groupby('diff_1_bin')['disc_1_abs'].transform(lambda x: x.quantile(percent_cut))
                cat_df['dim1_included'] = cat_df['disc_1_abs'] >= dim1_threshold
            final_df = pd.concat([final_df, cat_df], axis=0, ignore_index=True)

    else:
        df['diff_0_bin'] = pd.qcut(df['Difficulty 0'], num_bins, labels=False)
        df['diff_1_bin'] = pd.qcut(df['Difficulty 1'], num_bins, labels=False)
        df['disc_0_abs'] = df['Discrimination 0']
        df['disc_1_abs'] = df['Discrimination 1']
        if dim_0_neg != False:
            dim0_threshold = df.groupby('diff_0_bin')['disc_0_abs'].transform(lambda x: x.quantile(1 - percent_cut))
            df['dim0_included'] = df['disc_0_abs'] <= dim0_threshold
        else:
            dim0_threshold = df.groupby('diff_0_bin')['disc_0_abs'].transform(lambda x: x.quantile(percent_cut))
            df['dim0_included'] = df['disc_0_abs'] >= dim0_threshold
        if dim_1_neg != False:
            dim1_threshold = df.groupby('diff_1_bin')['disc_1_abs'].transform(lambda x: x.quantile(1 - percent_cut))
            df['dim1_included'] = df['disc_1_abs'] <= dim1_threshold
        else:
            dim1_threshold = df.groupby('diff_1_bin')['disc_1_abs'].transform(lambda x: x.quantile(percent_cut))
            df['dim1_included'] = df['disc_1_abs'] >= dim1_threshold
        final_df = df
    
    final_df.to_csv(f"{output_dir}{num_bins}_{percent_cut}_{data_filter}.csv")

    model_df = pd.read_csv(input_model, index_col="model")
    original_df = pd.read_csv(all_responses, index_col="question_id")
    original_df.columns = original_df.columns.str[9:]
    original_df["benchmark"] = read_df["benchmark"]
    final_df_dim0 = final_df[final_df["dim0_included"] == True]
    final_df_dim1 = final_df[final_df["dim1_included"] == True]
    final_df_dim0 = final_df_dim0.set_index("question_id")
    final_df_dim1 = final_df_dim1.set_index("question_id")
    original_df_dim0 = original_df[original_df.index.isin(final_df_dim0.index)]
    original_df_dim1 = original_df[original_df.index.isin(final_df_dim1.index)]

    original_df_dim0.to_csv(f"{output_dir}{num_bins}_{str(percent_cut)}_{data_filter}_original_df_dim0.csv")
    original_df_dim1.to_csv(f"{output_dir}{num_bins}_{str(percent_cut)}_{data_filter}_original_df_dim1.csv")
    
    ref_dim0 = model_df["Ability 0"]
    ref_dim1 = model_df["Ability 1"]


    original_df_dim0.groupby('benchmark').mean(numeric_only=True).to_csv(f"{output_dir}{num_bins}_{str(percent_cut)}_{data_filter}_original_df_dim0_averages.csv")
    original_df_dim1.groupby('benchmark').mean(numeric_only=True).to_csv(f"{output_dir}{num_bins}_{str(percent_cut)}_{data_filter}_original_df_dim1_averages.csv")

    dim0_bench = original_df_dim0.groupby('benchmark').mean(numeric_only=True)
    dim1_bench = original_df_dim1.groupby('benchmark').mean(numeric_only=True)
    orig_bench = original_df.groupby('benchmark').mean(numeric_only=True)

    dim0_bench.loc["Ability 0"] = ref_dim0
    dim0_bench.loc["Ability 1"] = ref_dim1

    dim1_bench.loc["Ability 0"] = ref_dim0
    dim1_bench.loc["Ability 1"] = ref_dim1

    orig_bench.loc["Ability 0"] = ref_dim0
    orig_bench.loc["Ability 1"] = ref_dim1

    dim0_bench.to_csv(f"{output_dir}{num_bins}_{str(percent_cut)}_{data_filter}_dim0_bench_averages.csv")
    dim1_bench.to_csv(f"{output_dir}{num_bins}_{str(percent_cut)}_{data_filter}_dim1_bench_averages.csv")

    # print(dim0_bench.head())
    # print(dim1_bench.head())

    final = []
    for bench in all_benchmarks.keys():
        try:
            corr_0_0, p_value_0_0 = pearsonr(dim0_bench.loc[bench], dim0_bench.loc["Ability 0"])
        except:
            corr_0_0, p_value_0_0 = "-", "-"
        try:
            corr_0_1, p_value_0_1 = pearsonr(dim0_bench.loc[bench], dim0_bench.loc["Ability 1"])
        except:
            corr_0_1, p_value_0_1 = "-", "-"

        try:
            corr_1_0, p_value_1_0 = pearsonr(dim1_bench.loc[bench], dim1_bench.loc["Ability 0"])
        except:
            corr_1_0, p_value_1_0 = "-", "-"
        try:
            corr_1_1, p_value_1_1 = pearsonr(dim1_bench.loc[bench], dim1_bench.loc["Ability 1"])
        except:
            corr_1_1, p_value_1_1 = "-", "-"
        try:
        # orig_bench[bench+"_dim0"] = dim0_bench.loc[bench]
            corr_0_self, p_0_self = pearsonr(dim0_bench.loc[bench], orig_bench.loc[bench])
            # print(dim0_bench.loc[bench])
            # print(orig_bench.loc[bench])
        except:
            corr_0_self, p_0_self = "-", "-"
        try:
        # orig_bench[bench+"_dim1"] = dim1_bench.loc[bench]
            corr_1_self, p_1_self = pearsonr(dim1_bench.loc[bench], orig_bench.loc[bench])
        except:
            corr_1_self, p_1_self = "-", "-"
        try:
            corr_no_0, p_value_no_0 = pearsonr(orig_bench.loc["Ability 0"], orig_bench.loc[bench])
        except:
            corr_no_0, p_value_no_0 = "-", "-"
        try:
            corr_no_1, p_value_no_1 = pearsonr(orig_bench.loc["Ability 1"], orig_bench.loc[bench])
        except:
            corr_no_1, p_value_no_1 = "-", "-"

        final.append({
            "Benchmark": bench,
            "All Prompts corr 0": corr_no_0,
            "All Prompts p 0": p_value_no_0,
            "All Prompts corr 1": corr_no_1,
            "All Prompts p 1": p_value_no_1,
            f"{percent_cut *100}% Filter Ability 0 corr 0": corr_0_0,
            f"{percent_cut *100}% Filter Ability 0 p 0": p_value_0_0,
            f"{percent_cut *100}% Filter Ability 0 corr 1": corr_0_1,
            f"{percent_cut *100}% Filter Ability 0 p 1": p_value_0_1,
            f"{percent_cut *100}% Filter Ability 1 corr 0": corr_1_0,
            f"{percent_cut *100}% Filter Ability 1 p 0": p_value_1_0,
            f"{percent_cut *100}% Filter Ability 1 corr 1": corr_1_1,
            f"{percent_cut *100}% Filter Ability 1 p 1": p_value_1_1,
            f"{percent_cut *100}% Filter Ability 0 corr self": corr_0_self,
            f"{percent_cut *100}% Filter Ability 0 p self": p_0_self,
            f"{percent_cut *100}% Filter Ability 1 corr self": corr_1_self,
            f"{percent_cut *100}% Filter Ability 1 p self": p_1_self,

        })
    df = pd.DataFrame(final)
    df.to_csv(f"{output_dir}{num_bins}_{str(percent_cut)}_{data_filter}_final.csv")