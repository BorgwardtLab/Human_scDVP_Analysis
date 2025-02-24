import argparse
import os, sys, pickle
import numpy as np
import pandas as pd
import warnings

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl

from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from utils import *


def main(args):

    # Read the results of the control model
    control_model_results = pd.read_csv(args.control_model_results, sep="\t").dropna(
        subset=["index"]
    )

    # Read the results of the treatment model
    treatment_model_results = pd.read_csv(
        args.treatment_model_results, sep="\t"
    ).dropna(subset=["index"])

    print("Control model results:")
    print(control_model_results.head())

    print("Treatment model results:")
    print(treatment_model_results.head())

    # Keep only proteins that are significant in the source)
    control_model_results = control_model_results[
        (control_model_results["q_value"] < args.pval_cutoff) & (np.abs(control_model_results["coefficient"]) > args.fold_change_cutoff)
    ]

    # How many proteins are in both models?
    common_proteins = np.intersect1d(
        control_model_results.index, treatment_model_results.index
    )
    print("Number of common proteins:", len(common_proteins))
    print(
        "Number of control only proteins:",
        len(set(control_model_results.index) - set(treatment_model_results.index)),
    )
    print(
        "Number of treatment only proteins:",
        len(set(treatment_model_results.index) - set(control_model_results.index)),
    )

    slope_comparison_results = pd.merge(
        control_model_results,
        treatment_model_results,
        on="index",
        suffixes=("_control", "_treated"),
    )

    print(slope_comparison_results.head())

    # Run a Wald test for all proteins
    slope_comparison_results["diff"] = (
        slope_comparison_results.coefficient_control
        - slope_comparison_results.coefficient_treated
    )

    if args.one_sided:
        abs_larger = np.abs(slope_comparison_results["coefficient_control"]) > np.abs(
            slope_comparison_results["coefficient_treated"]
        )
        slope_comparison_results.loc[
            abs_larger, "diff"
        ] = np.abs(slope_comparison_results.loc[
            abs_larger, "diff"
        ])
        slope_comparison_results.loc[
            ~abs_larger, "diff"
        ] = -np.abs(slope_comparison_results.loc[
            ~abs_larger, "diff"
        ])

    # Calculate the standard error of the difference
    slope_comparison_results["se_diff"] = np.sqrt(
        slope_comparison_results["bse_control"] ** 2
        + slope_comparison_results["bse_treated"] ** 2
    )

    # Calculate the z-statistic (or t-statistic if sample size is small)
    slope_comparison_results["z_stat"] = (
        slope_comparison_results["diff"] / slope_comparison_results["se_diff"]
    )

    # Calculate the p-value (two-tailed test)
    slope_comparison_results["p_value"] = 1 - stats.norm.cdf(
        np.abs(slope_comparison_results["z_stat"])
    )

    if args.one_sided:
        # We test the hypothesis that the absolute difference is greater than 0
        slope_comparison_results["p_value"] = slope_comparison_results["p_value"] / 2

    # Add multiple testing correction
    slope_comparison_results["q_value"] = multipletests(
        np.nan_to_num(slope_comparison_results["p_value"], 1.0), method="fdr_bh"
    )[1]
    slope_comparison_results["log2_fold_change"] = np.log2(
        slope_comparison_results["coefficient_control"]
        / slope_comparison_results["coefficient_treated"]
    )
    slope_comparison_results.sort_values("q_value", inplace=True)
    slope_comparison_results.reset_index(inplace=True, drop=True)
    slope_comparison_results.rename(columns={"index": "protein"}, inplace=True)
    slope_comparison_results = slope_comparison_results.dropna(subset=["protein"])

    print(slope_comparison_results.head())

    # Save results
    output = args.output
    if args.one_sided:
        output = output.replace(".tsv", "_one_sided.tsv")

    slope_comparison_results.to_csv(
        output,
        sep="\t",
        index=True,
        header=True,
    )

    # Create volcano plot
    if args.plot:

        slope_comparison_results.set_index("protein", inplace=True)

        if not args.one_sided:

            plt.figure(figsize=(10, 8))
            plt.scatter(
                slope_comparison_results["diff"],
                -np.log10(slope_comparison_results["q_value"]),
                color="grey",
                alpha=0.5,
            )

            plt.ylabel("-Log10 Q-value")
            plt.title("Volcano Plot")

            # Add labels for significant points
            significance_threshold = args.pval_cutoff
            delta_threshold = args.delta_cutoff

            sig_pos_points = slope_comparison_results[
                (slope_comparison_results["q_value"] < significance_threshold)
                & (slope_comparison_results["diff"] > delta_threshold)
            ]

            sig_neg_points = slope_comparison_results[
                (slope_comparison_results["q_value"] < significance_threshold)
                & (slope_comparison_results["diff"] < -delta_threshold)
            ]

            for idx, row in sig_pos_points.iterrows():
                plt.annotate(idx, (row["diff"], -np.log10(row["q_value"])))

            for idx, row in sig_neg_points.iterrows():
                plt.annotate(idx, (row["diff"], -np.log10(row["q_value"])))

            # Add significant scatter
            plt.scatter(
                sig_pos_points["diff"],
                -np.log10(sig_pos_points["q_value"]),
                alpha=0.5,
                color="red",
            )
            plt.scatter(
                sig_neg_points["diff"],
                -np.log10(sig_neg_points["q_value"]),
                alpha=0.5,
                color="blue",
            )

            # Add lines for thresholds
            plt.axhline(-np.log10(significance_threshold), color="red", linestyle="--")
            plt.axvline(delta_threshold, color="red", linestyle="--")
            plt.axvline(-delta_threshold, color="red", linestyle="--")
            plt.xlabel("Coefficient delta")

        else:

            fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)

            slope_comparison_results_cond = {
                "Central to Portal": slope_comparison_results.loc[
                    slope_comparison_results["coefficient_control"] > 0
                ],
                "Portal to Central": slope_comparison_results.loc[
                    slope_comparison_results["coefficient_control"] < 0
                ],
            }

            for ax, condition in zip(axes, ["Central to Portal", "Portal to Central"]):

                ax.scatter(
                    slope_comparison_results_cond[condition]["diff"],
                    -np.log10(slope_comparison_results_cond[condition]["q_value"]),
                    color="grey",
                    alpha=0.5,
                )

                # Add labels for significant points
                significance_threshold = args.pval_cutoff
                delta_threshold = args.delta_cutoff

                sig_pos_points = slope_comparison_results_cond[condition][
                    (
                        slope_comparison_results_cond[condition]["q_value"]
                        < significance_threshold
                    )
                    & (
                        slope_comparison_results_cond[condition]["diff"]
                        > delta_threshold
                    )
                ]

                for idx, row in sig_pos_points.iterrows():
                    ax.annotate(idx, (row["diff"], -np.log10(row["q_value"])))

                # Add significant scatter
                ax.scatter(
                    sig_pos_points["diff"],
                    -np.log10(sig_pos_points["q_value"]),
                    alpha=0.5,
                    color="red",
                )

                # Add lines for thresholds
                ax.axhline(
                    -np.log10(significance_threshold), color="red", linestyle="--"
                )
                ax.axvline(delta_threshold, color="red", linestyle="--")
                ax.set_xlim(0, 2)
                ax.set_title(condition)
                ax.set_xlabel("Control - treated (absolute)")
                ax.set_ylabel("-Log10 Q-value")

        plt.tight_layout()
        plt.savefig(
            output.replace("tsv", "png"),
        )
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run OLS on the scores")
    parser.add_argument(
        "--control_model_results",
        type=str,
        help="Path to the table containing results of the control model",
        required=True,
    )
    parser.add_argument(
        "--treatment_model_results",
        type=str,
        help="Path to the table containing results of the treatment model",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output file",
        default="slope_comparison_results.tsv",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the results",
    )
    parser.add_argument(
        "--pval_cutoff",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--fold_change_cutoff",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--delta_cutoff",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--one_sided",
        action="store_true",
        help="Whether to perform a one-sided test on the absolute difference of the coefficients",
    )
    args = parser.parse_args()
    main(args)
