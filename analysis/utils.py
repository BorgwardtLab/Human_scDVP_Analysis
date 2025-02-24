import mygene
import pandas as pd
from pybiomart import Server
from pyorthomap import findOrthologsMmHs


def mouse_to_human(gene_list):

    server = Server(host="http://www.ensembl.org")
    server.marts["ENSEMBL_MART_ENSEMBL"].list_datasets()

    map_df = findOrthologsMmHs(
        from_filters="external_gene_name", from_values=gene_list
    ).map()

    mouse_genes = map_df.external_gene_name.values
    human_genes = map_df.hgnc_symbol.values

    return {
        mouse: human
        for mouse, human in zip(mouse_genes, human_genes)
        if pd.notnull(human)
    }


def switch_uniprot_iD_to_Gene_Name(df, protein_col="protein", species="human"):
    # Get the list of UniProt IDs
    uniprot_ids = df[protein_col].tolist()

    # Remove the isoform information
    base_uniprot_ids = [x.split("-")[0] for x in uniprot_ids]

    seen_uniprot_ids = set()  # Track the UniProt IDs already seen

    # Retrieve gene symbols using MyGeneInfo
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        base_uniprot_ids, scopes="uniprot", fields="symbol", species=species
    )

    # Create a dictionary to store the mapping from UniProt IDs to gene symbols
    id_to_gene = {}
    for result in results:
        if "symbol" in result:
            gene_symbol = result["symbol"]
            if result["query"] in seen_uniprot_ids:
                print(
                    f"Duplicate UniProt ID: {result['query']}"
                )  # Handle duplicate UniProt ID
            else:
                id_to_gene[result["query"]] = gene_symbol
                seen_uniprot_ids.add(result["query"])
        else:
            id_to_gene[result["query"]] = result["query"]

    # Replace the UniProt IDs with gene symbols in the DataFrame
    df[protein_col] = [id_to_gene.get(x.split("-")[0], "") for x in uniprot_ids]
    return df


def remove_intensity_outliers(df, groups):
    """Implement outlier removal using Tukey's fences on protein intensity data.

    Args:
        - df (pd.DataFrame): DataFrame containing protein intensities where rows represent proteins and columns represent samples.
        - groups (pd.Series): Series containing group labels for each sample. The function will remove outliers per protein and per group.

    Returns:
        - pd.DataFrame: DataFrame with outliers removed.

    """

    # Function to remove outliers using Tukey's fences
    def remove_outliers(series):
        """
        Remove outliers from a Series using Tukey's fences.
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.where((series >= lower_bound) & (series <= upper_bound))

    # Transpose the filtered dataframe to align with group handling
    sample_names = df.columns
    df = df.T.copy()
    df["group"] = groups

    # Create a function to handle outlier removal for each group
    def process_group(group):
        # Process each numeric column independently
        numeric_data = group.drop(columns=["group"])
        filtered_data = numeric_data.apply(remove_outliers, axis=0)
        # Add the group column back
        filtered_data["group"] = group["group"]
        return filtered_data

    # Group by 'group' and process
    grouped = df.groupby("group", group_keys=False)
    processed = grouped.apply(process_group)

    # Transpose back to the original format and restore column names
    filtered_df = processed.drop(columns=["group"]).T
    filtered_df.columns = sample_names

    return filtered_df


def normalize_protein_intensities(df):
    """Normalize protein intensities in accordance with a detected core proteome.

    Args:
        - df (pd.DataFrame): DataFrame containing protein intensities where rows represent proteins and columns represent samples.

    Returns:

        - pd.DataFrame: Normalized DataFrame with protein intensities adjusted.

    This function performs the following steps:
    1. Identify Core Proteome Median Intensity: Identify proteins (rows) which are present in all samples (columns) and calculate the median intensity for each sample and then computes the overall median across these sample medians. Call this value proteome_core_median.
    2. Calculate normalization factors for each sample. The normalization factor for each sample is calculated by dividing the overall median protein intensity by the sample’s median protein intensity. This factor indicates how much each sample’s intensities need to be adjusted to align with the overall median.
    3. Apply the normalization factor to each intensity.

    """
    # Step 1: Identify Core Proteome Median Intensity
    core_proteome = df.dropna(axis=0, how="any")
    proteome_core_median = core_proteome.median().median()

    # Step 2: Calculate Normalization Factors for each sample
    sample_medians = core_proteome.median(axis=0)
    normalization_factors = proteome_core_median / sample_medians

    # Step 3: Apply Normalization Factor to each intensity
    normalized_df = df.apply(lambda x: x * normalization_factors[x.name], axis=0)

    return normalized_df


def filter_samples_by_protein_count(df):
    """
    Filters out columns (samples) in a DataFrame based on the number of identified proteins.

    A sample is excluded if the count of identified proteins (non-NaN entries) is:

        1) 1.5 standard deviations below the median count across all samples, or
        2) 3 standard deviations above the median count across all samples.

    This function is useful in proteomics data analysis, where the consistency in the number
    of identified proteins across samples is crucial for ensuring reliable comparisons and
    analyses.

    Args:
        - df (pandas.DataFrame): The DataFrame to be filtered, where rows represent proteins and columns represent samples.
    Returns:
        - pandas.DataFrame: A new DataFrame where samples with low or unusually high numbers of identified proteins have been excluded.

    """
    # Calculate the count of non-NaN entries (identified proteins) per sample
    protein_counts = df.count()

    # Calculate the median and standard deviation of these counts
    median_protein_count = protein_counts.median()
    std_dev_protein_count = protein_counts.std()

    # Define the thresholds for filtering
    lower_threshold = median_protein_count - 1.5 * std_dev_protein_count
    upper_threshold = median_protein_count + 3 * std_dev_protein_count

    # Filter and return the DataFrame
    return df.loc[
        :, (protein_counts >= lower_threshold) & (protein_counts <= upper_threshold)
    ]
