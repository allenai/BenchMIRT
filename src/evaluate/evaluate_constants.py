"""
Contains constants used in evaluation processes
"""
all_benchmarks = {
    "bbh": "reasoning",
    "bbq": "safety",
    "do_anything_now": "safety",
    "gpqa": "reasoning",
    "harmbench": "safety",
    "ifeval": "reasoning",
    "math": "reasoning",
    "mmlu-pro": "reasoning",
    "musr": "reasoning",
    "strongreject": "safety",
    "toxigen": "safety",
    "trustllm_jailbreaktrigger": "safety",
    "wildguardtest": "safety",
    "wildjailbreak": "safety",
    "wmdp": "safety",
    "xstest": "safety",
}


heldout_data_baselines_to_run = {
    "Multi_Dim_IRT_Total_Heldout": {
        "priors": "vague",
        "epochs": 5000,
        "dims": 2,
        "lr": 0.2,
        "lr_decay": 0.9999,
        "initializers": [],
        "baseline_type": "Multi_Dim_IRT_Total_Heldout"
    },
    "Multi_Dim_IRT_Total_Heldout_3_dim": {
        "priors": "vague",
        "epochs": 5000,
        "dims": 3,
        "lr": 0.2,
        "lr_decay": 0.9999,
        "initializers": [],
        "baseline_type": "Multi_Dim_IRT_Total_Heldout"
    },
    "Multi_Dim_IRT_Total_Heldout_4_dim": {
        "priors": "vague",
        "epochs": 5000,
        "dims": 4,
        "lr": 0.2,
        "lr_decay": 0.9999,
        "initializers": [],
        "baseline_type": "Multi_Dim_IRT_Total_Heldout"
    },
    "Single_Dim_IRT_Total_Heldout": {
        "priors": "vague",
        "epochs": 5000,
        "dims": 1,
        "lr": 0.2,
        "lr_decay": 0.9999,
        "initializers": [],
        "baseline_type": "Single_Dim_IRT_Total_Heldout"
    },
    "Single_Dim_IRT_Category_Heldout": {
        "priors": "vague",
        "epochs": 5000,
        "dims": 1,
        "lr": 0.2,
        "lr_decay": 0.9999,
        "initializers": [],
        "baseline_type": "Single_Dim_IRT_Category_Heldout"
    },
    "Model_Average_Total": {"baseline_type": "Model_Average_Total"},
    "Model_Average_Category": {"baseline_type": "Model_Average_Category"},
    "Model_Average_Benchmark": {"baseline_type": "Model_Average_Benchmark"}
}