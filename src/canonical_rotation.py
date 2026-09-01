#!/usr/bin/env python3

import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import procrustes
from typing import Tuple, Dict, Any
import os
import json
from glob import glob
from statsmodels.multivariate.factor_rotation import rotate_factors


class CanonicalRotator:
    def __init__(self, dims: int, reference_path: str = None):
        self.dims = dims
        self.reference_path = reference_path
        self.reference_gamma = None
        if self.reference_path and os.path.exists(self.reference_path):
            self.reference_gamma = np.load(self.reference_path)
    
    def canonical_init(self, gamma_std):
        # Oblique oblimin with gamma=0 (equivalently, quartimin). Returns
        # (L, T) with L = gamma_std @ inv(T.T), i.e. T is the rotation matrix.
        gamma_canonical, rotation = rotate_factors(gamma_std, "oblimin", 0, "oblique")
        return np.asarray(gamma_canonical), np.asarray(rotation)
    

    def rotate_params(self, params: Dict[str, list], save_reference: bool = False) -> Dict[str, Any]:

        theta_raw = np.array(params["ability"])
        b_raw = np.array(params["diff"])
        gamma_raw = np.array(params["disc"])
        

        # Standardize each matrix independently
        # theta_std = StandardScaler().fit_transform(theta_raw)
        # b_std = StandardScaler().fit_transform(b_raw)
        # gamma_std = StandardScaler().fit_transform(gamma_raw)
        theta_std = theta_raw
        b_std = b_raw
        gamma_std = gamma_raw
        if self.reference_gamma is None:
            # First run: "canonical" initialization (i ended up using oblimin actually for a reference gamma that is more interpretable) 
            gamma_rot, R = self.canonical_init(gamma_std)
            self.reference_gamma = gamma_rot  # Set reference for future runs
            print("No reference gamma found. Using canonically-rotated gamma as reference.")
        else:
            print("Aligning to reference gamma using Procrustes...")
            # Align to reference by finding the optimal rotation matrix R
            H = gamma_std.T @ self.reference_gamma
            U, _, Vt = np.linalg.svd(H)
            R = U @ Vt
            
            # Apply the rotation
            gamma_rot = gamma_std @ R

            init_disparity = procrustes(self.reference_gamma, gamma_std)[2]  # Initial distance before Procrustes

            # Calculate disparity for with scipy's procrustes
            disparity = procrustes(self.reference_gamma, gamma_rot)[2]  

            #procrustes distances don't really change because they are invariant to orthogonal transformations, but we can check correlation before and after rotation
      
        # Rotate all matrices consistently
        theta_rot = theta_std @ R.T
        b_rot = b_std @ R.T
        

        result = {
            "ability_rot": theta_rot.tolist(),
            "diff_rot": b_rot.tolist(),
            "disc_rot": gamma_rot.tolist(),
            "disc": gamma_std.tolist(),
            "procrustes_disparity": float(disparity) if "disparity" in locals() else np.nan,
            "initial_procrustes_disparity": float(init_disparity) if 'init_disparity' in locals() else None,
            "initial_pearsonr": float(pearsonr(self.reference_gamma.flatten(), gamma_std.flatten())[0]) if 'init_disparity' in locals() else np.nan,
            "pearsonr": float(pearsonr(self.reference_gamma.flatten(), gamma_rot.flatten())[0]) if 'gamma_rot' in locals() else np.nan,
            "dims": self.dims
        }
        
        # Save reference
        if save_reference or self.reference_gamma is None:
            ref_path = self.reference_path or "irt_reference_gamma.npy"
            np.save(ref_path, gamma_rot)
            result["reference_saved"] = ref_path
        
        return result



def compare_runs(files):
    N_items = 15000
    print("Reference file:", files[0])
    ref_params = json.load(open(files[0], "r"))
    item_indices = random.sample(range(len(ref_params["disc"])), N_items)  
    ref_params_sampled = {
        "ability": np.array(ref_params["ability"]),
        "diff": np.array(ref_params["diff"])[item_indices],
        "disc": np.array(ref_params["disc"])[item_indices]
    }
    rotator = CanonicalRotator(dims=2)
    rotated1 = rotator.rotate_params(ref_params_sampled, save_reference=True)
    gamma = np.array(rotated1['disc_rot'])

    for file in files:
        with open(file, "r") as f:
            data = json.load(f)

        data_sampled = {
            "ability": np.array(data["ability"]),
            "diff": np.array(data["diff"])[item_indices],
            "disc": np.array(data["disc"])[item_indices]
        }
        
        rotated2 = rotator.rotate_params(data_sampled, save_reference=False)
        gamma_rot2 = np.array(rotated2['disc_rot'])
        print(f"File: {file}, Gamma shape: {gamma_rot2.shape}")
        # print("Initial Procrustes distance between reference and unrotated gamma: {:.4f}".format(rotated2['initial_procrustes_disparity']))
        # print(f"Procrustes distance between reference and {file}: {rotated2['procrustes_disparity']:.4f}")
        print(f"Initial Pearson correlation between reference and {file}: {rotated2['initial_pearsonr']:.4f}")
        print(f"Pearson correlation between reference and {file}: {rotated2['pearsonr']:.4f}")

        # print(f"Comparing initial gammas for {file}:")
        for dim in range(rotator.dims):
            corr_init = np.corrcoef(
                [gamma[dim] for gamma in rotator.reference_gamma],
                [gamma[dim] for gamma in rotated2['disc']]
            )[0, 1]
            corr_later = np.corrcoef(
                [gamma[dim] for gamma in rotator.reference_gamma],
                [gamma[dim] for gamma in gamma_rot2]
            )[0, 1]
            print(f"Gamma correlation for dim {dim} between reference and {file}: {corr_init:.4f}")
            print(f"Gamma correlation for dim {dim} between reference and {file}: {corr_later:.4f}")
            print("***")

        #print(f"Procrustes distance between reference and {file}: {procrustes(gamma_rot1, gamma_rot2)[2]:.4f}")
        print("-" * 50)

def rotate_all_runs(files):
    rotator = CanonicalRotator(dims=2)
    ref_file = random.choice(files)  # Randomly select a reference file
    with open(ref_file, "r") as f:
        ref_data = json.load(f)
    ref_params = {
        "ability": np.array(ref_data["ability"]),
        "diff": np.array(ref_data["diff"]),
        "disc": np.array(ref_data["disc"])
    }
    rotator.rotate_params(ref_params, save_reference=True)  # Set reference using the first file

    for file in files:
        with open(file, "r") as f:
            data = json.load(f)

        data_sampled = {
            "ability": np.array(data["ability"]),
            "diff": np.array(data["diff"]),
            "disc": np.array(data["disc"])
        }
        
        rotated = rotator.rotate_params(data_sampled, save_reference=False)
        # Save the rotated parameters back to a new JSON file
        output_file = file.replace(".json", "_rotated.json")
        with open(output_file, "w") as f_out:
            json.dump(rotated, f_out, indent=4)
        print(f"Rotated parameters saved to {output_file}")

if __name__ == "__main__":

    # files = glob("safety-reasoning-benchmark/rotational_invariance/runs/*.json")
    files = glob("runs/*.json")
    print(files)
    random.shuffle(files)
    compare_runs(files)

    # To rotate all runs and save new JSON files with rotated parameters, uncomment the following line:
    rotate_all_runs(files)
    
