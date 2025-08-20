import mrcfile
import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Callable
from scipy.spatial.transform import Rotation
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

def testfunc():
    print("YEOOO. this is a TEST!")

def default_filepattern(in_path):
    return in_path.split("_")[2]

def make_datadict(
        dir_path: os.PathLike | str, 
        protein_of_interest: str, 
        image_filepattern: Callable = default_filepattern
        ) -> dict:
    """Reads in refine-template cav files from a directory and indexes them by their image

    Parameters:
    -----------
    dir_path: os.PathLike or str
    Path of the directory that the differential refine-template data is located in.

    protein_of_interest: str
    name of the protein of intsrest; this name will be used to find within the filepaths.
    
    image_filepattern: function
    Function that inputs a filepattern and returns an image number. should work on basenames
    Used to index the dictionary.

    Returns
    -------
    dict: 
    dictionary with dataframes produced from indexing files by image path
    """
    filedict = {}
    for file in os.listdir(dir_path):
        if file == ".DS_Store":
            continue
        try:
            image = image_filepattern(file)
        except FileNotFoundError as e:
            print(e)
            print(f"skipping....")
            continue

        if image not in filedict.keys():
            filedict[image] = {
                "control": None, 
                protein_of_interest: None
                }
        if protein_of_interest in file:
            filedict[image][protein_of_interest] = pd.read_csv(os.path.join(dir_path, file), index_col=[0])
        if "control" in file:
            filedict[image]["control"] = pd.read_csv(os.path.join(dir_path, file), index_col=[0])
    return filedict


def create_match_stats(
        df1: pd.DataFrame, 
        df2: pd.DataFrame
        ) -> pd.DataFrame:
    """Computes a matched df by particle index, with rotation distance, defocus distance, and spatial distance
    
    Parameters:
    -----------
    df1: 
    The dataframe to match 
    df2: 
    The dataframe to be matched to df1
    Returns: 
    match_stats: pd.DataFrame
    The dataframe with stats in it.
    """
    rot1 = Rotation.from_euler(
        "xzy",
        df1[["refined_phi", "refined_psi", "refined_theta"]].to_numpy(),
        degrees=True
        )
    rot2 = Rotation.from_euler(
        "xzy",
        df2[["refined_phi", "refined_psi", "refined_theta"]].to_numpy(),
        degrees=True
        )
    cols = ["refined_pos_x", "refined_pos_y"]
    rot_distance = (rot2 * rot1.inv()).magnitude() * (180 / np.pi)
    euc_distance = np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)
    defocus_distance = np.abs(df1["refined_relative_defocus"].values - df2["refined_relative_defocus"].values)
    match_stats = pd.DataFrame(
        data=np.column_stack([
            df1["particle_index"],
            euc_distance, 
            rot_distance, 
            defocus_distance
            ]), 
        columns=[
            "particle_index",
            "euclidean_distance", 
            "rotation_distance", 
            "defocus_difference"
            ])
    return match_stats

def match_and_compute_log2s(
        datadict: dict
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produces match statistics and log2 ratios by particle index and image for each file.

    Parameters:
    -----------
    datadict: dict
    dictionary of the control and poi refine-template dataframes, indexed by image

    Returns: 
    --------
    match_stats: pd.DataFrame
    DataFrame with euclidean distances, defocus differences, and rotation distances 
    for each peak, indexed by particle_index and image
    log2_ratios: pd.DataFrame
    DataFrame with log2 ratios of refined_scaled_mips, indexed by particle_index and image
    """
    dfs = []
    log2s_dfs = []
    for image, data in datadict.items():
        assert len(data) == 2, f"Expected 2 entries per image, got {len(data)} for {image}"
        try:
            control = data["control"]
        except KeyError:
            raise KeyError(f"No 'control' dataframe found for image {image}")
        [(poi, poi_df)] = [(k, v) for k, v in data.items() if k != "control"]
        tmp_match_stats = create_match_stats(control, poi_df)
        if tmp_match_stats.empty:
            print(f"Image {image} has no peaks! skipping...")
            continue
        tmp_match_stats["particle_index"] = tmp_match_stats["particle_index"].values.astype(int)
        tmp_match_stats["image"] = image
        log2_arr = np.log2(poi_df["refined_scaled_mip"].values / control["refined_scaled_mip"].values)
        log2_df = pd.DataFrame(
            data = np.column_stack([
                control["particle_index"],
                [image] * control.shape[0],
                log2_arr
            ]
            ),
            columns=[
                "particle_index", 
                "image", 
                f"log2({poi}/60S)"
                ]
        )
        log2_df[f"log2({poi}/60S)"] = log2_df[f"log2({poi}/60S)"].values.astype(float)

        log2s_dfs.append(log2_df)
        dfs.append(tmp_match_stats)

    match_stats = pd.concat(dfs)
    log2_ratios = pd.concat(log2s_dfs)
    return match_stats, log2_ratios


def plot_match_stats(
        match_stats: pd.DataFrame,
        title: str,
        ) -> None:
    """Plots histograms from match_stats dataframe

    Parameters:
    -----------
    match_stats: pd.DataFrame
    DataFrame that the plot is based on. should have Rotaton distance, 
    euclidean distance, and absolute defocus distance

    title: str
    What to title the plot

    Returns:
    --------
    None
    """
    axes_label_dict = {"size": 20}
    titledict = {"size": 25}
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(17, 6))
    fig.suptitle(f"Match Stats for {title}", fontsize=35, fontweight="bold")
    axs = axs.flatten()
    for ax in axs:
        ax.tick_params(axis="both", labelsize=15)

    # euclidean distance
    axs[0].set_title("Euclidean\ndistance histogram", **titledict)
    axs[0].hist(
        match_stats["euclidean_distance"], 
        50, 
        label=f"Peaks: (n = {match_stats.shape[0]})", 
        color="teal")
    axs[0].set_ylabel("Count", **axes_label_dict)
    axs[0].legend(loc="upper right", frameon=False, prop={'size': 20})
    axs[0].set_xlabel("Euclidean distance (pxls)", **axes_label_dict)

    # rotation diatance
    axs[1].set_title("Rotation\ndistance histogram", **titledict)
    axs[1].hist(
        match_stats["rotation_distance"],
        50, 
        color="magenta")
    axs[1].set_ylabel("Count", **axes_label_dict)
    axs[1].set_xlabel("Rotation distance (degrees)", **axes_label_dict)

    axs[2].set_title("Absolute Defocus\ndistance histogram", **titledict)
    axs[2].hist(
        match_stats["defocus_difference"], 
        50, 
        color="tab:green")
    axs[2].set_ylabel("Count", **axes_label_dict)
    axs[2].set_xlabel("Defocus difference (\u212b)", **axes_label_dict)
    plt.tight_layout()
    plt.show()

def plot_initial_ratios(
        log2_ratios: pd.DataFrame, 
        title: str) -> None:
    """Plot the initial log2 ratios in a pretty plot with a kde
    
    Parameters:
    ----------
    log2_ratios: pd.DataFrame
    DataFrame with the log2 ratio information in it

    title: str
    extra info for the title

    Returns:
    --------
    None
    """
    coi = [col for col in log2_ratios.columns if col.startswith("log2")][0]
    poi = coi.split("(")[1].split("/")[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(
        log2_ratios[coi].values, 
        bins=200, 
        color="lightgray",
        density=True, 
        label=f"Peaks (n = {log2_ratios.shape[0]})"
        )
    sns.kdeplot(
        log2_ratios[coi].values, 
        ax=ax, 
        fill=True, 
        color="violet", 
        label="Kde estimate")
    ax.legend(loc="upper right", frameon=True, prop={"size": 13})
    ax.set_xlabel(coi, fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.set_title(f"log2 ratios for {poi} {title}", fontsize=20)
    plt.show()




def select_best_gmm(
        log2_data: pd.DataFrame, 
        title: str,
        max_components: int = 5, 
        ) -> None:
    """
    Fits GMMs with 1 to max_components and selects the best one based on BIC and AIC.

    Parameters:
    -----------
    log2_data: pd.DataFrame
        DataFrame with log2 ratios in it

    title: str
        Title for the plot

    max_components: int, default=5
        Max number of components to test.
    """
    # Reshape data for sklearn (n_samples, n_features)
    coi = [col for col in log2_data.columns if col.startswith("log2")][0]
    data = log2_data[coi].values.reshape(-1, 1)
    
    aics, bics, models = [], [], []

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=23)
        gmm.fit(data)
        aics.append(gmm.aic(data))
        bics.append(gmm.bic(data))
        models.append(gmm)
    
    best_index_bic = np.argmin(bics)
    best_index_aic = np.argmin(aics)

    # Plot AIC and BIC curves
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, max_components + 1), aics, marker='o', label="AIC")
    plt.plot(range(1, max_components + 1), bics, marker='s', label="BIC")
    plt.xlabel("Number of GMM Components")
    plt.ylabel("Score")
    plt.title(f"GMM Model Selection (AIC vs BIC): {title}")
    plt.xticks(range(1, max_components + 1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Best model by BIC: {best_index_bic + 1} components (BIC = {bics[best_index_bic]:.2f})")
    print(f"Best model by BIC: {best_index_aic + 1} components (AIC = {aics[best_index_aic]:.2f})")

def fit_gmms_and_likelihoods(
        log2_data: pd.DataFrame, 
        title: str
        ) -> pd.DataFrame:
    """Fits a 2-component Gaussian mixture model to the data, plots, and returns the responsilities
    
    Parameters:
    -----------
    log2_data: pd.DataFrame
    DataFrame containing log2s for each image and particle index

    Returns:
    --------
    log2s_likelihoods: pd.DataFrame
    log2 ratio dataframe with likelihoods included

    """
    coi = [col for col in log2_data.columns if col.startswith("log2")][0]
    log2_arr = log2_data[coi].values
    data = log2_arr.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2,
        random_state=23
    )
    gmm.fit(data)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs = axs.flatten()
    axs[0].hist(data, bins=50, density=True, alpha=0.5, color="gray")


    x = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    pdf = np.exp(logprob)
    axs[0].plot(x, pdf, "k--", label="Mixture PDF")
    # individual components
    for i, (mean, cov, weight) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
        component_pdf = (
            weight
            * (1 / np.sqrt(2 * np.pi * cov.item()))
            * np.exp(-0.5 * ((x - mean.item()) ** 2) / cov.item())
        )
        axs[0].plot(x, component_pdf, label=f"Component {i+1}")
    axs[0].set_xlabel(coi)
    axs[0].set_ylabel("Density")
    axs[0].legend()
    axs[0].set_box_aspect(1)
    responsibilities = gmm.predict_proba(data).T
    axs[1].scatter(log2_arr, responsibilities[0], s=5, label="Component 1")
    axs[1].scatter(log2_arr, responsibilities[1], s=5, label="Component 2")
    axs[1].set_xlabel(coi)
    axs[1].set_ylabel("Likelihood")
    axs[1].legend(frameon=False)
    axs[1].set_box_aspect(1)
    fig.suptitle(f"2 component GMM fit and likelihood {title}", fontweight="bold", fontsize=20)
    plt.tight_layout()
    plt.show()

    log2s_likelihoods = log2_data.copy()
    log2s_likelihoods["comp_1_likelihood"] = responsibilities[0]
    log2s_likelihoods["comp_2_likelihood"] = responsibilities[1]
    return log2s_likelihoods

def filter_log2_ratios(
        log2s: pd.DataFrame, 
        component: int,
        log2_threshold: float = 0.0, 
        likelihood_threshold: float = 0.95,
        ) -> pd.DataFrame:
    """Filters log2 ratios by likelihood and by log2 ratio
    
    Parameters:
    -----------
    log2s: pd.DataFrame
    DataFrame containing log2_ratios and likelihoods
    component: int
    log2_threshold: float
    Log2 ratio above which to include log2 ratios. default 0.0
    likelihood_threshold: float
    Likelihood above which to include log2 ratios. default 0.95

    Returns:
    --------
    filtered_log2s: pd.DataFrame
    log2s filtered according to the above thresholds
    """
    coi = [col for col in log2s.columns if col.startswith("log2")][0]
    filtered_log2s = log2s[
        (log2s[f"comp_{component}_likelihood"] >= likelihood_threshold) & 
        (log2s[coi] >= log2_threshold)].copy()
    print(f"{(filtered_log2s.shape[0] / log2s.shape[0]) * 100:.2F}% of peaks ({filtered_log2s.shape[0]}) have the protein of interest!")
    return filtered_log2s

def load_constrained_search_paths(
        dir_name: os.PathLike | str, 
        axes: list[tuple[str, str]],
        image_filepattern: Callable
        ) -> dict[str, dict[tuple[str, str], pd.DataFrame]]:
    """Filters constrained search files by axis and creates dict by-image of the dataframes

    Parameters:
    -----------
    dir_name: str
    Path to the directory to search over
    axes: list(tuple(str, str))
    list of axes to search over, in the format (from, to)
    image_filepattern: Callable
    function that takes in a in filepath and outputs the image name
    
    Returns: 
    --------
    datadict: dict
    dictionary containing the dataframes of the constrained-search results, organized by image  
    """
    datadict = {}
    for file in os.listdir(dir_name):
        if file.endswith("results_above_threshold.csv") == False:
            continue
        conf_pdb = file.split("_")[1]
        axis_pdb = file.split("relto_")[1].split("_cs")[0]
        image = image_filepattern(file)
        axis = (conf_pdb, axis_pdb)
        if axis in axes:
            if image not in datadict.keys():
                datadict[image] = {}
            datadict[image][axis] = pd.read_csv(os.path.join(dir_name, file), index_col=[0])
    return datadict

def determine_axis_overlap(
        data_40S: dict[str, dict[tuple[str, str], pd.DataFrame]], 
        axes: list[tuple[str, str]]
        ) -> dict[str, pd.DataFrame]:
    """Takes input datadict and adds a column depending on which one has the highest log2 ratio
    
    Only works with 
    Parameters:
    -----------
    data_40S: dict
    by_image, by_axis dict of constrained_dataframes
    axes: list
    list of axes tuples to iterate over, must be 2 long

    Returns:
    --------
    by_image: dict
    dictionary by image of a single dataframe, with a new column best-axis and the correct columns 
    per best axis
    """
    assert len(axes) == 2, "This function only works for 2 axes"
    by_image = {}
    for image, value in data_40S.items():
        ax1 = value[axes[0]]
        ax2 = value[axes[1]]
        pi1 = set(ax1["particle_index"])
        pi2 = set(ax2["particle_index"])
        matched = pi1 & pi2
        pi1_only = pi1 - pi2
        pi2_only = pi2 - pi1
        ax1_best = ax1[ax1["particle_index"].isin(pi1_only)].copy()
        ax2_best = ax2[ax2["particle_index"].isin(pi2_only)].copy()
        merged = pd.merge(
            ax1[ax1["particle_index"].isin(matched)],
            ax2[ax2["particle_index"].isin(matched)],
            on="particle_index",
            suffixes=("_ax1", "_ax2")
        )
        ax1_wins = merged["refined_scaled_mip_ax1"] > merged["refined_scaled_mip_ax2"]
        # Keep ax1 version of rows where ax1 wins
        ax1_matched = merged.loc[ax1_wins, ["particle_index"] + [c for c in merged.columns if c.endswith("_ax1")]].copy()
        ax1_matched.columns = ax1.columns  # strip suffixes to match ax1

        # Keep ax2 version of rows where ax2 wins
        ax2_matched = merged.loc[~ax1_wins, ["particle_index"] + [c for c in merged.columns if c.endswith("_ax2")]].copy()
        ax2_matched.columns = ax2.columns  # strip suffixes to match ax2
        new_ax1 = pd.concat([
            ax1_best, ax1_matched
        ])
        new_ax2 = pd.concat([
            ax2_best, ax2_matched
        ])
        new_ax1["best_axis"] = [axes[0]] * new_ax1.shape[0]
        new_ax2["best_axis"] = [axes[1]] * new_ax2.shape[0]
        df = pd.concat([new_ax1, new_ax2])
        by_image[image] = df.sort_values(by="particle_index")

    return by_image
    


def match_poi_to_40S(
        filtered_poi: pd.DataFrame, 
        datadict_40S: dict[str, pd.DataFrame], 
        peaks_60S: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
    """Matches protein of interest peaks to 40S rotation.

    Matches protein of interest peaks to 40S "rotation state" (axis-wise), 
    and returns dataframe with log2 ratio and rotation-state
    
    Parameters:
    -----------
    filtered_poi: pd.DataFrame
    protein of interest log2 dataframe
    datadict_40S: 
    by-image, by_axis dataframe of 40S constrained search results
    peaks_60S: dict 
    by-image dictionary of 60S refine-template results
    
    Returns: 
    --------
    out_df:
    dataframe with 60S peaks + eEF2 info + 40S axis info
    """
    dfs_40S = []
    dfs_60S = []
    for image, value in datadict_40S.items():
        value["index_combo"] = [(str(ind), image) for ind in value["particle_index"].values]
        dfs_40S.append(value)
        df = peaks_60S[image]
        df["index_combo"] = [(str(ind), image) for ind in df["particle_index"].values]
        dfs_60S.append(df)

    all_images_40S = pd.concat(dfs_40S)
    all_images_60S = pd.concat(dfs_60S)
    # add: best_axis, poi_present, original_offset_phi
    filtered_poi["index_combo"] = filtered_poi.apply(lambda r: (r["particle_index"], r["image"]), axis=1)    
    out_df = all_images_60S.copy(deep=True)

    # mark eEF2 presence
    out_df["contains_eEF2"] = out_df["index_combo"].isin(filtered_poi["index_combo"])

    # merge in axis + rotation_state from 40S data
    out_df = out_df.merge(
        all_images_40S[["index_combo", "best_axis", "original_offset_phi"]],
        on="index_combo",
        how="left"   # keeps all rows in out_df
    )

    # rename for clarity
    out_df = out_df.rename(columns={"original_offset_phi": "rotation_state"})
    out_df = out_df.where(pd.notnull(out_df), False)
    return out_df

def make_autopct(values: list) -> Callable:
    """Generates text to put on the pie chart with absolute numbers and percents
    
    Parameters:
    -----------
    values: list
    the values used to produce percents

    Returns:
    --------
    my_autopct: Callable
    function that will use the percents passed into the pie chart and display absolute numbers
    """
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}% ({v:d})'.format(p=pct,v=val)
    return my_autopct

def plot_overlaps(
        all_data_df: pd.DataFrame, 
        title: str
        ) -> None:
    """Plots pie charts of the overlap between eEF2 and 40S.

    Parameters:
    -----------
    all_data_df: pd.DataFrame
    DataFrame that contains the (index, image) combo and a best-axis and contains_eEF2 column
    title: str
    What to title the plot

    Returns:
    --------
    None
    """
    all_peaks = set(all_data_df["index_combo"])
    all_40S = set(all_data_df[all_data_df["best_axis"] != False]["index_combo"])
    all_eEF2 = set(all_data_df[all_data_df["contains_eEF2"] != False]["index_combo"])
    eEF2_and_40S = all_eEF2 & all_40S
    eEF2_no_40S = all_eEF2 - all_40S
    no_eEF2_40S = all_40S - all_eEF2
    no_nothing = all_peaks - (eEF2_and_40S | eEF2_no_40S | no_eEF2_40S)
    colors = ["tab:blue", "tab:green", "tab:red", "lightgray"]

    p = [
        len(no_eEF2_40S),
        len(eEF2_and_40S), 
        len(eEF2_no_40S), 
        len(no_nothing)
            ]
    labs = ["40S w/o eEF2", "eEF2 w/ 40S", "eEF2 w/o 40S","60S only"]
    fig, ax = plt.subplots()
    ax.pie(
        p, 
        startangle=90,
        labels=labs, 
        colors=colors,
        autopct=make_autopct(p)
        )
    ax.axis("equal")
    ax.set_title(f"eEF2 comparison pie chart for {title}", fontsize=20, fontweight='bold')
    plt.show()


