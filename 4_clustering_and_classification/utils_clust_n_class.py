import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw
import math
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, NMF
#import umap
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from hdbscan import HDBSCAN

# ------------------------------------------------------------------------------
# Function to load a pickle file as a DataFrame
def load_pickle_as_df(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)  # Pandas DataFrame


# ------------------------------------------------------------------------------
# Function to save the merged DataFrame as a pickle file
def save_pickle(df, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(df, f)
        

# ------------------------------------------------------------------------------
def pca_cumulative_variance_plot(merged_df, norm_name, tilesize, n_comp=50):
    
    pca = PCA(n_components=n_comp)  # default is 50
    pca_components = pca.fit_transform(merged_df)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_comp + 1), pca.explained_variance_ratio_.cumsum(), marker='.')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA Variance Explained for {tilesize} µm from {norm_name}')
    return plt


# ------------------------------------------------------------------------------
def pca_single_cumulative_variance_plot(sample, pkl_dict, norm_name, tilesize, n_comp=50):

    pca = PCA(n_components=n_comp)  # default is 50
    pca_components = pca.fit_transform(pkl_dict[sample])

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_comp + 1), pca.explained_variance_ratio_.cumsum(), marker='.')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    if sample.startswith("satac"):
        plt.title(f'PCA Variance Explained for {tilesize} µm from sATAC for {norm_name}')
    else:
        plt.title(f'PCA Variance Explained for {tilesize} µm from ST for {norm_name}')
    return plt


# ------------------------------------------------------------------------------
def all_dim_red(merged_df, normalisation_method, colors_dict_satac_visium, tilesize, selected_methods=None, n_comp=2, random_seed=123):
    '''
    Function for plotting some dimensionality reduction approaches for the merged dataframes and color by spatial method.
    Params:
    - merged_df: Pandas.DataFrame, the merged DataFrame for the two spatial methods
    - normalisation_method: String, the merged DataFrame for the two spatial methods
    - colors_dict_satac_visium: Pandas.DataFrame, the merged DataFrame for the two spatial methods
    - tilesize: int, size of the tiles or spot
    - n_comp = int, number of components for the dimensionality reduction
    - selected_methods: List of strings, specifying which methods to apply (default: all)
    '''
    # Define a dictionary of available models
    available_models = {
        "PCA": PCA(n_components=n_comp),
        "SVD": TruncatedSVD(n_components=n_comp, random_state=random_seed),
        "ICA": FastICA(n_components=n_comp, random_state=random_seed),
        "NMF": NMF(n_components=n_comp, random_state = random_seed),
        "UMAP": umap.UMAP(n_components=n_comp, random_state=random_seed),
        "tSNE": TSNE(n_components=n_comp, perplexity=50, random_state=random_seed) # Uncomment if needed
    }
    
    # Filter models based on selected_methods
    if selected_methods is None:
        models = list(available_models.items())
    else:
        models = [(name, available_models[name]) for name in selected_methods if name in available_models]
    
    # Initialize a figure to organize subplots
    fig, axes = plt.subplots(1, len(models), figsize=(len(models) * 5, 5))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    dim_red_results = {}

    # Create a binary color mapping based on the presence of "-" in index names
    index_names = merged_df.index.astype(str)
    color_labels = index_names.str.contains("-", regex=False).astype(int)  # 1 if "-" present, 0 otherwise

    # Define colors for the two classes
    color_map = {1: colors_dict_satac_visium["Visium ST"], 0: colors_dict_satac_visium["Spatial ATAC"]}
    colors = [color_map[label] for label in color_labels]

    for i, (method_name, model) in enumerate(models):
        # creating a temporal variable for modification of the dataframe
        temp_merged_df = merged_df
        
        # adaption for NMF
        if method_name == "NMF":
            min_value = merged_df.min().min()
            if min_value < 0:
                temp_merged_df += abs(min_value)

        
        # Model transformation
        result = model.fit_transform(temp_merged_df)
        
        dim_red_results[method_name] = result

        # Plot the reduced result in a subplot
        ax = axes[i]  
        scatter = ax.scatter(result[:, 0], result[:, 1], c=colors, s=4, cmap='coolwarm')

        ax.set_title(f"{method_name} of all features for {tilesize} µm from {normalisation_method}")
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')

    # Create legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, 
                            markerfacecolor=color, markersize=9)
                    for label, color in colors_dict_satac_visium.items()]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
            ncol=2, fontsize='medium')

    plt.tight_layout()
    return fig, dim_red_results


# ------------------------------------------------------------------------------
def single_dim_red(single_df, selected_methods=None, n_comp=50, random_seed=123):
    '''
    Function for plotting some dimensionality reduction approaches for the merged dataframes and color by spatial method.
    Params:
    - single_df: Pandas.DataFrame, the merged DataFrame for the two spatial methods
    - n_comp = int, number of components for the dimensionality reduction
    - selected_methods: List of strings, specifying which methods to apply (default: all)
    '''
    # Define a dictionary of available models
    available_models = {
        "PCA": PCA(n_components=n_comp),
        "SVD": TruncatedSVD(n_components=n_comp, random_state=random_seed),
        "ICA": FastICA(n_components=n_comp, random_state=random_seed),
        "NMF": NMF(n_components=n_comp, random_state = random_seed)#,
        # "UMAP": umap.UMAP(n_components=n_comp, random_state=random_seed),
        # "tSNE": TSNE(n_components=n_comp, perplexity=50, random_state=random_seed) # Uncomment if needed
    }
    
    # Filter models based on selected_methods
    if selected_methods is None:
        models = list(available_models.items())
    else:
        models = [(name, available_models[name]) for name in selected_methods if name in available_models]


    for i, (method_name, model) in enumerate(models):
        # creating a temporal variable for modification of the dataframe
        temp_merged_df = single_df
        
        # adaption for NMF
        if method_name == "NMF":
            min_value = single_df.min().min()
            if min_value < 0:
                temp_merged_df += abs(min_value)

        
        # Model transformation
        result = model.fit_transform(temp_merged_df)
        
    return result


# # ------------------------------------------------------------------------------
# def single_clustering_by_method(selected_components, n_clust, TILE_SIZE, selected_methods):
#     available_clustering_methods = [
#         ("K-Means", KMeans(n_clusters=n_clust, random_state=123)),
#         # ("Spectral Clustering", SpectralClustering(n_clusters=n_clust, affinity='nearest_neighbors', n_neighbors=10, 
#         #                                         eigen_solver='arpack', random_state=123)),
#         ("MeanShift", MeanShift()),  # bandwidth will be estimated later
#         ("DBSCAN", DBSCAN(eps=0.5, min_samples=20)),
#         ("HDBSCAN", HDBSCAN(min_cluster_size=15)),
#         ("Agglomerative", AgglomerativeClustering(n_clusters=n_clust, metric='euclidean', linkage='ward')),
#         ("Affinity Propagation", AffinityPropagation(preference=-50)),  # Adjust preference as needed
#         ("Birch", Birch(threshold=0.5, n_clusters=n_clust))#,  # Adjust threshold as needed
#         # ("Gaussian Mixture", GaussianMixture(n_components=5, random_state=123))
#     ]
    
#     # Filter models based on selected_methods
#     if selected_methods is None:
#         methods = list(available_clustering_methods.items())
#     else:
#         methods = [(name, available_clustering_methods[name]) for name in selected_methods if name in available_clustering_methods]
    
#     # Initialize a figure to organize subplots
#     fig, axes = plt.subplots(1, len(selected_methods), figsize=(len(selected_methods) * 6, 6))
#     axes = axes.flatten()  # Flatten the axes array for easy indexing

#     clustering_results = {}
        
#     for i, (method_name, model) in enumerate(methods):
#         # Estimate bandwidth if using MeanShift
#         if method_name == "MeanShift":
#             est_band = estimate_bandwidth(selected_components)
#             model.set_params(bandwidth=est_band)
        
#         # Fit the clustering model and predict labels
#         cluster_labels = model.fit_predict(selected_components)
        
#         clustering_results[method_name] = cluster_labels
        
#         # Plot in the appropriate subplot
#         ax = axes[i]  
#         scatter = ax.scatter(selected_components[:, 0], selected_components[:, 1], c=cluster_labels, s=5, cmap='spectral')
#         ax.set_title(f"{method_name} Clustering - {TILE_SIZE}um tiles")
#         ax.set_xlabel(f'Component 1')
#         ax.set_ylabel(f'Component 2')
    
#     plt.tight_layout()
#     plt.show()

#     return fig, clustering_results


# ----------------------------------------------------------------------------
def single_clustering_by_method(sample_name, selected_components, n_clust, TILE_SIZE, selected_methods=None):
    available_clustering_methods = {
        "K-Means": KMeans(n_clusters=n_clust, random_state=123),
        "HDBSCAN": HDBSCAN(min_cluster_size=15),
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clust, linkage='ward')
    }
    
    if selected_methods is None:
        methods = available_clustering_methods
    else:
        methods = {name: available_clustering_methods[name] for name in selected_methods if name in available_clustering_methods}

    clustering_results = {}

    fig, axes = plt.subplots(1, len(methods), figsize=(len(methods) * 6, 6))
    if len(methods) == 1:
        axes = [axes]  # Ensure it's iterable for a single method

    for i, (method_name, model) in enumerate(methods.items()):
        # Fit and predict clusters
        cluster_labels = model.fit_predict(selected_components.values)
        # Assign colors to clusters using Seaborn's Husl colormap
        unique_labels = np.unique(cluster_labels)
        colors = sns.color_palette("husl", len(unique_labels))
        label_color_map = {label: colors[idx] for idx, label in enumerate(unique_labels)}
        color_list = np.array([label_color_map[label] for label in cluster_labels])

        # # Create merged DataFrame with cluster labels & colors
        # merged_df = pd.DataFrame(
        #     cluster_labels, index=selected_components.index, columns=["Cluster"]
        # )
        # merged_df[["R", "G", "B"]] = color_list  # Store RGB color values
        
        temp = pd.merge(pd.DataFrame(cluster_labels, index=selected_components.index, columns=["Cluster"]), 
                             pd.DataFrame(color_list, index=selected_components.index, columns=["R", "G", "B"]), 
                            left_index = True, right_index=True)
        
        # Store results
        # clustering_results[method_name] = merged_df
        clustering_results[method_name] = temp

        # Plot results
        ax = axes[i]
        scatter = ax.scatter(selected_components.iloc[:, 0], selected_components.iloc[:, 1], 
                             c=color_list, s=5)
        ax.set_title(f"{method_name} Clustering - {TILE_SIZE}µm tiles")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    if "satac" in sample_name:
        fig.suptitle('Clustering for the sATAC sample', fontsize=16)
    else:
        fig.suptitle('Clustering for the ST sample', fontsize=16)
    
    plt.tight_layout()
    plt.show()

    return fig, clustering_results


# ----------------------------------------------------------------------------
def just_draw_shapes_on_satac(row, image_to_draw, px_size, shape="rectangle"):
    if row['in_tissue'] and not any(pd.isna(val) for val in [row["R"], row["G"], row["B"]]):
        x_pxl, y_pxl = row["pxl_row"], row["pxl_col"]
        color = tuple((row[["R", "G", "B"]].astype(float) * 255).astype(int))  # Convert to proper RGB scale
        color_fill = (color[0], color[1], color[2])
        
        if shape == "rectangle":
            # Compute tile boundaries
            left = x_pxl - math.floor(px_size / 2)
            upper = y_pxl - math.floor(px_size / 2)
            right = x_pxl + math.floor(px_size / 2)
            lower = y_pxl + math.floor(px_size / 2)

            image_to_draw.rectangle([left, upper, right, lower], outline=color, width=80)
            # image_to_draw.rectangle([left, upper, right, lower], fill=color_fill, outline=color, width=80)
        elif shape == "circle":
            #image_to_draw.circle([x_pxl, y_pxl], radius = px_size/2, outline=color, width=30)
            image_to_draw.circle([x_pxl, y_pxl], radius = px_size/2, fill=color_fill, outline=color, width=2)


# ----------------------------------------------------------------------------
def just_draw_shapes_on_visium(row, image_to_draw, px_size, shape="rectangle"):
    if row['in_tissue'] and not any(pd.isna(val) for val in [row["R"], row["G"], row["B"]]):
        x_pxl, y_pxl = row["pxl_col"], row["pxl_row"]
        color = tuple((row[["R", "G", "B"]].astype(float) * 255).astype(int))  # Convert to proper RGB scale
        color_fill = (color[0], color[1], color[2])

        if shape == "rectangle":
            # Compute tile boundaries
            left = x_pxl - math.floor(px_size / 2)
            upper = y_pxl - math.floor(px_size / 2)
            right = x_pxl + math.floor(px_size / 2)
            lower = y_pxl + math.floor(px_size / 2)

            image_to_draw.rectangle([left, upper, right, lower], outline=color, width=40)
            # image_to_draw.rectangle([left, upper, right, lower], fill=color_fill, outline=color, width=30)
        elif shape == "circle":
            #image_to_draw.circle([x_pxl, y_pxl], radius = px_size/2, outline=color, width=30)
            image_to_draw.circle([x_pxl, y_pxl], radius = px_size/2, fill=color_fill, outline=color, width=2)


# ----------------------------------------------------------------------------
def draw_tiles_on_wsi(wsi_image, coords_df, clust_n_colors_dict, vis_or_atac, um_size, normalisation_name, 
                      shape_to_plot = "rectangle" , atac_px_size = None, visium_px_size = None, saving_path=None):
    """
    Draws rectangles on the whole slide image (WSI) based on the spot coordinates.

    :param wsi_image: PIL Image object of the full-resolution WSI
    :param coords_df: DataFrame containing spot coordinates with 'pxl_col' and 'pxl_row'
    :param px_size: Size of each tile in pixels
    :param saving_path: (Optional) Path to save the new WSI with drawn rectangles
    :return: Image with drawn rectangles
    """
    
    # initializing the figure. Number of plot = number of methods in the input dataframe
    fig, axes = plt.subplots(1, len(clust_n_colors_dict), figsize=(len(clust_n_colors_dict) * 10, 10))
    
    for i, (clustering_name, df_cluster_colors) in enumerate(clust_n_colors_dict.items()):
        
        # hard copying the WSI, otherwise the draws would be placed one on top of the other
        copied_wsi = wsi_image.copy()
        draw = ImageDraw.Draw(copied_wsi)
        
        # hard copying the dataframes, otherwise they would undergo modifications
        temp_coords_df = coords_df.copy()
        temp_clust_df = df_cluster_colors.copy()
        
        # converting the RGB columns to the correct RGB format (from [0,1] to [0,255])
        # temp_clust_df[["R", "G", "B"]] = ((temp_clust_df[["R", "G", "B"]].astype(float) * 255).astype(int))
        
        # creating temporal column with just barcodes
        temp_clust_df.insert(0, "just_barcodes", [i.split("_")[0] for i in temp_clust_df.index.tolist()])
        
        # merging on that temporal column
        merged_df = pd.merge(temp_coords_df, temp_clust_df, 
                            left_on="barcode", right_on="just_barcodes", 
                            how="outer")
        
        #merged_df[['R', 'G', 'B']] = (merged_df[['R', 'G', 'B']].astype(float) * 255).astype(int)

        #merged_df
        #print(merged_df[['in_tissue','R', 'G', 'B']].head())
    
        if len(temp_clust_df) == 1:
            axes = [axes]  # Ensure it's iterable for a single method

        # looping over clustering method
        
        # Draw all the rectangles
        if vis_or_atac == "sATAC" and not visium_px_size:
            # merged_df.apply(just_draw_rectangles_on_satac, axis=1, args =(draw, atac_px_size))
            merged_df.apply(just_draw_shapes_on_satac, axis=1, args=(draw, atac_px_size, shape_to_plot))
        
        elif vis_or_atac == "visium" and visium_px_size:
            # merged_df.apply(just_draw_rectangles_on_visium, axis=1, args=(draw, visium_px_size))
            merged_df.apply(just_draw_shapes_on_visium, axis=1, args=(draw, visium_px_size, shape_to_plot))
        
        else:
            "Correctly define all the parameters for Visium or sATAC. Help yourself with the function documentation."
            return
        
        
        # resizing the image for simplicity in loading step
        # defining the size of the image
        height_in_pixels = 2000
        width_in_pixels = round((copied_wsi.size[0] * height_in_pixels / copied_wsi.size[1]))
        new_size = width_in_pixels, height_in_pixels
        
        print(f"Original size: {copied_wsi.size}, Rescaled size: {new_size}")
        high_res_image = copied_wsi.resize(new_size, Image.Resampling.LANCZOS)
        
        
        # Plot results
        ax = axes[i]
        ax.imshow(high_res_image)
        ax.set_title(f"Results from {clustering_name} clustering - {um_size}µm tiles", fontsize=16)
        ax.axis("off")
        
        # Extract unique clusters and colors for legend
        unique_clusters = temp_clust_df.drop_duplicates(subset=["Cluster"])[["Cluster", "R", "G", "B"]]
        cluster_labels = [f"Cluster {int(c)}" for c in unique_clusters["Cluster"]]
        cluster_colors = [(r, g, b) for r, g, b in zip(unique_clusters["R"], unique_clusters["G"], unique_clusters["B"])]
    
    # Add a horizontal legend
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in sorted(zip(cluster_labels, cluster_colors), key = lambda x: x[0])]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(cluster_labels), fontsize=12) # bbox_to_anchor=(0.5, 0.97),
    # Add a horizontal legend
    # legend_patches = [mpatches.Patch(color=[row.R/255, row.G/255, row.B/255], label=f"Cluster {int(row.cluster)}") 
    #                    for _, row in df_cluster_colors.iterrows()]
    # fig.legend(handles=legend_patches, loc="lower center", ncol=len(df_cluster_colors), fontsize=12)
    
        
    if vis_or_atac == "visium":
        fig.suptitle(f'Clustering for the {vis_or_atac.capitalize()} {normalisation_name} sample', fontsize=22, fontweight="bold", y=1.05)
    else:
        fig.suptitle(f'Clustering for the {vis_or_atac} {normalisation_name} sample', fontsize=22, fontweight="bold", y=1.05)

    
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make room for legend
    plt.show()

    # eventually, saving
    if saving_path:    
        fig.savefig(saving_path, format="PDF", bbox_inches="tight")
    
    return fig


# ----------------------------------------------------------------------------
def saving_obtained_clusters(coords_df, df_cluster_colors, saving_path):
    """
    Saves a merged dataframe containing spatial coordinates and cluster assignments with colors.

    :param coords_df: DataFrame containing spot coordinates with a 'barcode' column.
    :param clust_n_colors_dict: Dictionary where keys are clustering names and values are DataFrames with cluster info, 
                                including 'R', 'G', and 'B' values.
    :param vis_or_atac: String, either "sATAC" or "visium", specifying the dataset type.
    :param um_size: Unused in the function but kept for consistency.
    :param normalisation_name: Unused in the function but kept for consistency.
    :param saving_path: Path where the final merged dataframe should be saved.
    """
            
    # Hard copy to avoid modifying the original dataframes
    temp_coords_df = coords_df.copy()
    temp_clust_df = df_cluster_colors.copy()
    
    # Convert RGB values from [0,1] range to [0,255] range
    temp_clust_df[["R", "G", "B"]] = (temp_clust_df[["R", "G", "B"]].astype(float) * 255).astype(int)
    
    # Extract barcode without the suffix
    temp_clust_df.insert(0, "just_barcodes", [i.split("_")[0] for i in temp_clust_df.index.tolist()])
    # and with the suffix
    temp_clust_df.insert(0, "image_name", [i for i in temp_clust_df.index.tolist()])
    
    # Merge with spatial coordinates dataframe
    merged_df = pd.merge(temp_coords_df, temp_clust_df, 
                            left_on="barcode", right_on="just_barcodes", 
                            how="outer")
    
    merged_df.drop(columns = ["just_barcodes"], inplace=True)
    
    # Save the merged dataframe to CSV, ensuring headers are included
    print("Saving at:\n", saving_path)
    merged_df.to_csv(saving_path, index=True, header=True)
    
    
