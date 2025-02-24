import argparse
import os, sys, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels

import warnings
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from utils import *

warnings.simplefilter("ignore", ConvergenceWarning)


def main(args):

    # Define groups
    group_dict = {
        1: [
            "NML4",
            "NML5",
            "NML6",
            "NML7",
            "NML10",
            "NML17",
            "NML20",
            "NML31",
            "NML9",
            "NML11",
            "NML15",
            "NML16",
            "CHTL3",
            "CHTL73",
        ],  # Controls
        2: ["CHTL36", "CHTL38", "CHTL59", "CHTL34"],  # Cases
        3: ["m3B", "m5C", "m4A"],  # mouse data
    }

    intensity = pd.read_csv(
        os.path.join(args.data_path, "scDVP_filtered.tsv"), sep="\t"
    )

    # intensity = switch_uniprot_iD_to_Gene_Name(intensity) # Rename proteins
    intensity = intensity[
        ["protein"]
        + [
            col
            for col in intensity.columns
            if any(
                patient + "_" in col for patient in group_dict[int(args.patient_group)]
            )
        ]
    ]

    # Filter proteins with too many missing values
    missing_values = intensity.isnull().sum(axis=1) / intensity.shape[1]
    intensity = intensity.loc[missing_values <= args.missing_threshold]

    print(filter_samples_by_protein_count(intensity.iloc[:, 1:]))

    # Filter samples with too few or too many proteins
    intensity = filter_samples_by_protein_count(intensity)

    print("Using a missing threshold of", args.missing_threshold)
    print("Keeping", intensity.shape[0], "proteins")

    # Normalize to the median intensity per cell
    intensity.iloc[:, 1:] = normalize_protein_intensities(intensity.iloc[:, 1:])

    # Remove outliers using Tukey's fences
    intensity.iloc[:, 1:] = remove_intensity_outliers(
        intensity.iloc[:, 1:],
        groups=intensity.iloc[:, 1:].columns.str.split("_").str[0],
    )

    # Normalize the rows using RobustScaler
    from sklearn.preprocessing import RobustScaler

    intensity.iloc[:, 1:] = (
        RobustScaler().fit_transform(intensity.iloc[:, 1:].values.T).T
    )

    # Read scores
    scores = pd.read_csv(os.path.join(args.data_path, "all_scores.tsv"), sep="\t")
    intensity = intensity.drop(
        columns=[col for col in intensity.columns[1:] if col not in scores.columns]
    )
    filtered_scores = scores.drop(
        columns=[col for col in scores if col not in intensity.columns]
    )

    filtered_scores = filtered_scores[intensity.columns[1:]]  # reorder columns
    filtered_scores.loc[2] = [x.split("_")[0] for x in filtered_scores.columns]

    # Fit LMM for each protein
    models = {}
    results = {}
    for protein in intensity["protein"]:
        # Get the row for this protein
        protein_data = intensity[intensity["protein"] == protein].iloc[0]

        # Extract the shape values for this protein
        y = protein_data[1:].astype(float).values

        # Fit LMM
        data = pd.DataFrame(
            {
                "y": y,
                "score": filtered_scores.loc[1].astype(float),
                "patient": filtered_scores.loc[2],
            }
        )

        # Fit mixed linear model
        model = smf.mixedlm(
            "y ~ score",
            data=data,
            groups=data["patient"],
            missing="drop",
        )
        fit_results = model.fit(method=["powell", "lbfgs"])
        models[protein] = fit_results

        # Store the results
        results[protein] = {
            "coefficient": fit_results.fe_params["score"],
            "p_value": fit_results.pvalues["score"],
            "random_intercept_variance": fit_results.cov_re.iloc[0, 0],
            "residual_variance": fit_results.scale,
            "log_likelihood": fit_results.llf,
            "aic": fit_results.aic,
            "bic": fit_results.bic,
            "bse": fit_results.bse["score"],
        }

    # Convert results to a DataFrame
    results_df = pd.DataFrame.from_dict(results, orient="index")

    # Correct multiple testing
    results_df["q_value"] = statsmodels.stats.multitest.multipletests(
        np.nan_to_num(results_df["p_value"], 1.0), method="fdr_bh"
    )[1]

    # Get protein SYMBOL names
    results_df.index = switch_uniprot_iD_to_Gene_Name(
        results_df.reset_index(),
        protein_col="index",
        species=args.species,
    )["index"]

    print(results_df.head())

    # Save results
    if args.save:

        # Save models
        with open(
            f"models_{args.patient_group}_cutoff={1 - args.missing_threshold}.pkl", "wb"
        ) as f:
            pickle.dump(models, f)

        # Save results
        results_df.to_csv(
            f"results_{args.patient_group}_cutoff={1 - args.missing_threshold}.tsv",
            sep="\t",
            index=True,
            header=True,
        )

        # Save results with ortholog names
        if args.species == "mouse":
            ortholog_map = mouse_to_human(results_df.index)
            renamed_results_df = results_df.loc[ortholog_map.keys()]
            renamed_results_df.index = [
                ortholog_map[x] for x in renamed_results_df.index
            ]
            unmapped_proteins = set(results_df.index) - set(ortholog_map.keys())

            pd.Series(list(unmapped_proteins)).to_csv(
                f"unmapped_proteins_{args.patient_group}_cutoff={1 - args.missing_threshold}.tsv",
                sep="\t",
                index=True,
                header=True,
            )
            renamed_results_df.to_csv(
                f"results_{args.patient_group}_cutoff={1 - args.missing_threshold}_human_names.tsv",
                sep="\t",
                index=True,
                header=True,
            )

    # Create volcano plot
    if args.plot:

        plt.figure(figsize=(10, 8))
        plt.scatter(
            results_df["coefficient"],
            -np.log10(results_df["q_value"]),
            color="grey",
            alpha=0.5,
        )

        plt.xlabel("Score coefficient")
        plt.ylabel("-Log10 Q-value")
        plt.title("Volcano Plot")

        # Add labels for significant points
        significance_threshold = args.pval_cutoff
        coeff_threshold = args.coeff_cutoff

        sig_pos_points = results_df[
            (results_df["q_value"] < significance_threshold)
            & (results_df["coefficient"] > coeff_threshold)
        ]

        sig_neg_points = results_df[
            (results_df["q_value"] < significance_threshold)
            & (results_df["coefficient"] < -coeff_threshold)
        ]

        for idx, row in (
            sig_pos_points.sort_values("q_value").head(args.annotate_top).iterrows()
        ):
            plt.annotate(idx, (row["coefficient"], -np.log10(row["q_value"])))

        for idx, row in (
            sig_neg_points.sort_values("q_value").head(args.annotate_top).iterrows()
        ):
            plt.annotate(idx, (row["coefficient"], -np.log10(row["q_value"])))

        # Add significant scatter
        plt.scatter(
            sig_pos_points["coefficient"],
            -np.log10(sig_pos_points["q_value"]),
            alpha=0.5,
            color="red",
        )
        plt.scatter(
            sig_neg_points["coefficient"],
            -np.log10(sig_neg_points["q_value"]),
            alpha=0.5,
            color="blue",
        )

        # Add lines for thresholds
        plt.axhline(-np.log10(significance_threshold), color="red", linestyle="--")
        plt.axvline(-coeff_threshold, color="red", linestyle="--")
        plt.axvline(coeff_threshold, color="red", linestyle="--")
        plt.xlim(-7.5, 7.5)

        plt.tight_layout()
        plt.savefig(
            f"volcano_plot_{args.patient_group}_cutoff={1 - args.missing_threshold}.png"
        )
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run LMM on the scores")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the directory containing the data",
        default=".",
    )
    parser.add_argument(
        "--patient_group",
        type=int,
        help="Name of the patient to use in the analysis",
        required=True,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the results",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the results",
    )
    parser.add_argument(
        "--missing_threshold",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--pval_cutoff",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--coeff_cutoff",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--species",
        type=str,
        default="human",
    )
    parser.add_argument(
        "--annotate_top",
        type=int,
        default=10,
    )

    args = parser.parse_args()
    main(args)
