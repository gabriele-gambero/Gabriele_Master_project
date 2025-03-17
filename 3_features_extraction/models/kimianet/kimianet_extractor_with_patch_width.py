# environment name: example_kimianet_11_8

# %%
import os
import tensorflow as tf
import subprocess

# %%

import regex
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob, pickle, skimage.io, pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# %%
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

# %% [markdown]
# Checking if the first chosen GPU was, in fact, selected.

# %%
# List physical devices available (CPU, GPU)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

for gpu in gpus:
    print("Device Name:", gpu.name)
    print("Device Type:", gpu.device_type)


# %%
if gpus:
    # Set only GPU:1 to be visible to TensorFlow
    try:
        tf.config.set_visible_devices(gpus[1], 'GPU')  # Adjusting to GPU 1. It maybe be set on 0 depending on the visible GPU that we've selected
        tf.config.experimental.set_memory_growth(gpus[1], True)  # Enable memory growth
    except RuntimeError as e:
        print(f"Error setting GPU configuration: {e}")

# Verify that only GPU:1 is used
print("Configured devices:", tf.config.get_visible_devices())


# %% [markdown]
# # 1.- Configuring variables

# %% [markdown]
# ## 1.1 - Setting the working path

# %%
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/3_features_extraction/")
print(os.getcwd())

# %% [markdown]
# Model name.

# %%
model = "kimianet"

# %% [markdown]
# Model weights download link:
# https://github.com/KimiaLabMayo/KimiaNet/blob/main/KimiaNet_Weights/weights/KimiaNetKerasWeights.h5
# 
# Or, with code:

# %%
import requests

# Define the URL of the raw file
url = "https://github.com/KimiaLabMayo/KimiaNet/raw/main/KimiaNet_Weights/weights/KimiaNetKerasWeights.h5"
# Define the file path where it will be saved
file_path = "./models/kimianet/KimiaNetKerasWeights.h5"

# Send a GET request to download the file
response = requests.get(url, stream=True)

# Save the file to the specified path
if response.status_code == 200:
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded successfully to {file_path}")
else:
    print(f"Failed to download. Status code: {response.status_code}")


# %% [markdown]
# Importing the summary files for both the techniques and the tiles dimension.\
# These contain the path to all the tiles folder, the original and the normalised ones.

# %%
# SUMMARY_SATAC_100um = "/disk2/work/gabgam/gigi_env/the_project/2_image_normalisation/output/satac_C1/v3_allspots/tiles_100/final_summary_for_all_100um_normalised_tiles.csv"
# SUMMARY_SATAC_68um = "/disk2/work/gabgam/gigi_env/the_project/2_image_normalisation/output/satac_C1/v3_allspots/tiles_68/final_summary_for_all_68um_normalised_tiles.csv"

# SUMMARY_VISIUM_100um = "/disk2/work/gabgam/gigi_env/the_project/2_image_normalisation/output/visium_2022_FF_WG_10X/img_not_changed_allspots/tiles_100/final_summary_for_all_100um_normalised_tiles.csv"
# SUMMARY_VISIUM_68um = "/disk2/work/gabgam/gigi_env/the_project/2_image_normalisation/output/visium_2022_FF_WG_10X/img_not_changed_allspots/tiles_68/final_summary_for_all_68um_normalised_tiles.csv"
# %%
SUMMARY_SATAC_100um = "/disk2/work/gabgam/gigi_env/the_project/2_image_normalisation/output/satac_C1/v3_allspots/tiles_100/final_summary_for_all_100um_normalised_tiles.csv"
SUMMARY_SATAC_68um = "/disk2/work/gabgam/gigi_env/the_project/2_image_normalisation/output/satac_C1/v3_allspots/tiles_68/final_summary_for_all_68um_normalised_tiles.csv"

SUMMARY_VISIUM_100um = "/disk2/work/gabgam/gigi_env/the_project/2_image_normalisation/output/visium_FFPE_dcis_idc_10X/img_not_changed_allspots/tiles_100/final_summary_for_all_100um_normalised_tiles.csv"
SUMMARY_VISIUM_68um = "/disk2/work/gabgam/gigi_env/the_project/2_image_normalisation/output/visium_FFPE_dcis_idc_10X/img_not_changed_allspots/tiles_68/final_summary_for_all_68um_normalised_tiles.csv"

# %% [markdown]
# Extracting sample name and coordinates systems from files name.

# %%
# for sATAC
temp_satac_names = SUMMARY_SATAC_100um.split('output/')[1].split('/tiles')[0].split("/")

SAMPLE_SATAC = temp_satac_names[0]
IMAGE_VERSION_AND_COORDINATE_SYSTEM_SATAC = temp_satac_names[1]

# for Visium
temp_visium_names = SUMMARY_VISIUM_100um.split('output/')[1].split('/tiles')[0].split("/")

SAMPLE_VISIUM = temp_visium_names[0]
IMAGE_VERSION_AND_COORDINATE_SYSTEM_VISIUM = temp_visium_names[1]


print(f"Processing for:\nsATAC sample: {SAMPLE_SATAC}\nsATAC coordinates system: {IMAGE_VERSION_AND_COORDINATE_SYSTEM_SATAC}\n\nVisium sample: {SAMPLE_VISIUM}\nVisium coordinates system: {IMAGE_VERSION_AND_COORDINATE_SYSTEM_VISIUM}")

# %% [markdown]
# Choosing the reference and the name system based on `TARGET_IS_<filename>`.

# %%
# path to the target image
PATH_TO_REFERENCE = "../2_image_normalisation/reference_images/reference_full.jpeg"
# `TARGET_IS_<filename>`
TARGET_IS = "target_is_reference_full"

# %% [markdown]
# Creating a dictionary for the path to the file, but with the reference image already defined.

# %%
complete_dict_summaries = {f"{SAMPLE_SATAC}_&_{IMAGE_VERSION_AND_COORDINATE_SYSTEM_SATAC}_&_{TARGET_IS}_100um": SUMMARY_SATAC_100um,
                  f"{SAMPLE_SATAC}_&_{IMAGE_VERSION_AND_COORDINATE_SYSTEM_SATAC}_&_{TARGET_IS}_68um": SUMMARY_SATAC_68um,
                  f"{SAMPLE_VISIUM}_&_{IMAGE_VERSION_AND_COORDINATE_SYSTEM_VISIUM}_&_{TARGET_IS}_100um": SUMMARY_VISIUM_100um,
                  f"{SAMPLE_VISIUM}_&_{IMAGE_VERSION_AND_COORDINATE_SYSTEM_VISIUM}_&_{TARGET_IS}_68um": SUMMARY_VISIUM_68um}
complete_dict_summaries = {name: pd.read_csv(path, index_col = [0]) for name, path in complete_dict_summaries.items()}

# %% [markdown]
# Specify the normalisation methods that have been chosen and the `ORIGINAL WSI` with the exact name in the summaries.

# %%
SELECTED_METHODS = ["ORIGINAL WSI", "fromWSI_histomicsTK_macenko_nomasking", "histomicsTK_macenko_nomasking", "stainNET"]

# %% [markdown]
# Choosing the weights.

# %%
NETWORK_WEIGHTS_ADDRESS = f"models/{model}/KimiaNetKerasWeights.h5"

# %% [markdown]
# Setting the output path.

# %%
comparison_of_analyses = f"{SAMPLE_SATAC}_{IMAGE_VERSION_AND_COORDINATE_SYSTEM_SATAC}_&_{SAMPLE_VISIUM}_{IMAGE_VERSION_AND_COORDINATE_SYSTEM_VISIUM}"
features_output_path = f"output/{model}/{comparison_of_analyses}/"

os.makedirs(features_output_path, exist_ok = True)

# %% [markdown]
# # 2. - Preprocessing and Feature Extraction Functions

# %%
summary_satac_100um = pd.read_csv(SUMMARY_SATAC_100um, index_col= [0])

# %% [markdown]
# Let's import the summary files with the paths to the folders and select the paths to the tiles folders for the ORIGINAL WSI and the methods that worked out.

# %%
filter_summaries_by_method = {name: df[df['Normalisation Method'].isin(SELECTED_METHODS)] for name, df in complete_dict_summaries.items()}

# %% [markdown]
# Selecting by the specified normalisation methods and the target.

# %%
filter_summaries_by_method = {name: df[df['Normalisation Method'].isin(SELECTED_METHODS) & df["Target"].isin(["-", TARGET_IS])].reset_index() for name, df in complete_dict_summaries.items()}


# %% [markdown]
# These are the KimiaNet features extraction functions.\

# %%
def preprocessing_fn(input_batch, network_input_patch_width):
    org_input_size = tf.shape(input_batch)[1]
    scaled_input_batch = tf.cast(input_batch, 'float32') / 255.0  # Ensuring dtype is float32
    resized_input_batch = tf.cond(tf.equal(org_input_size, network_input_patch_width),
                                  lambda: scaled_input_batch, 
                                  lambda: tf.image.resize(scaled_input_batch, 
                                                          (network_input_patch_width, network_input_patch_width)))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_format = "channels_last"
    mean_tensor = tf.constant(-np.array(mean), dtype=tf.float32)  # Ensuring mean tensor is float32
    standardized_input_batch = tf.keras.backend.bias_add(resized_input_batch, mean_tensor, data_format)
    standardized_input_batch /= std
    return standardized_input_batch


# ------------------------------------------------------------------------------------------------------
def kimianet_feature_extractor(network_input_patch_width, weights_address):
    if not os.path.exists(weights_address):
        raise FileNotFoundError(f"Weights file not found at {weights_address}")
    
    dnx = DenseNet121(include_top=False, weights=weights_address, 
                      input_shape=(network_input_patch_width, network_input_patch_width, 3), pooling='avg')
    kn_feature_extractor = Model(inputs=dnx.input, outputs=GlobalAveragePooling2D()(dnx.layers[-3].output))
    kn_feature_extractor_seq = Sequential([Lambda(preprocessing_fn, 
                                                  arguments={'network_input_patch_width': network_input_patch_width}, 
                                                  input_shape=(None, None, 3), dtype=tf.uint8)])
    kn_feature_extractor_seq.add(kn_feature_extractor)
    return kn_feature_extractor_seq


# ------------------------------------------------------------------------------------------------------
def extract_features(patch_dir, network_weights_address, 
                     network_input_patch_width, batch_size, img_format):
    patch_dir = row["Path to folder"]
    feature_extractor = kimianet_feature_extractor(network_input_patch_width, network_weights_address)
    patch_adr_list = [pathlib.Path(x) for x in glob.glob(os.path.join(patch_dir, f'*.{img_format}'))]
    
    # Debug: Print number of files found
    print(f"Number of files found: {len(patch_adr_list)}")
    
    if len(patch_adr_list) == 0:
        print("No files found. Please check the patch directory and image format.")
        return
    
    feature_dict = {}

    for batch_st_ind in tqdm(range(0, len(patch_adr_list), batch_size)):
        batch_end_ind = min(batch_st_ind + batch_size, len(patch_adr_list))
        batch_patch_adr_list = patch_adr_list[batch_st_ind:batch_end_ind]
        
        # Debug: Print current batch size
        print(f"Processing batch from index {batch_st_ind} to {batch_end_ind}")
        
        patch_batch = np.array([skimage.io.imread(str(x)) for x in batch_patch_adr_list])
        batch_features = feature_extractor.predict(patch_batch)
        feature_dict.update(dict(zip([x.stem for x in batch_patch_adr_list], list(batch_features))))
        
        # with open(extracted_features_save_adr, 'wb') as output_file:
        #     pickle.dump(feature_dict, output_file, pickle.HIGHEST_PROTOCOL)
    
    print("Feature extraction completed.")
    return pd.DataFrame.from_dict(feature_dict)


# %% [markdown]
# Let's perform the real features extraction step.
# 
# Let's set the batch size (accordingly to GPU computational power), the image file format (tipically `.jpg`).\
# The real extraction function will be called inside the dictionary where each key will contain all the info regarding the images in the folder (sample origin, size in micrometers and the eventual normalisation method) and the relative value will be the dataframe originated from the function which will contain all the extracted features per image (images indeces have their own real name).
# 
# In the end, all the feature dataframes will be saved as `.pickle` files in the `./output` folder.

# %%
batch_size = 16 #32
img_format = 'jpg'
network_input_patch_width = 1000 #224 #1000


all_extracted_features_from_selected_summaries = {}

# Create a list of (name, row) pairs
entries = [
    (name, row) for name, df in filter_summaries_by_method.items() for _, row in df.iterrows()
]

# Iterate over the list
for name, row in entries:
    key = f"{name}_{row['Normalisation Method']}"  # Using dictionary-like access
    try:
        features = extract_features(row, NETWORK_WEIGHTS_ADDRESS, network_input_patch_width, batch_size, img_format)
        all_extracted_features_from_selected_summaries[key] = features # features as dataframes
    except Exception as e:
        print(f"Error processing {key}: {e}")

# the gpu_timer warning at the end of the output doesn't affect it. It's a bug related to the versions of some packages like nightly and CUDNN:
# https://github.com/tensorflow/tensorflow/issues/71791

# parameters_folder = f"batchsize{batch_size}" if not network_input_patch_width else f"batchsize{batch_size}_inputpathwidth{network_input_patch_width}"  # might be useless

# saving all the dataframes as .pickles files with the specific name from the folder and the normalisation method that was used.
for name, features_df in all_extracted_features_from_selected_summaries.items():
    if network_input_patch_width:
        with open(os.path.join(features_output_path, f"{name}_width{network_input_patch_width}.pickle"), 'wb') as output_file:
            pickle.dump(features_df, output_file, pickle.HIGHEST_PROTOCOL)

    else:
        with open(os.path.join(features_output_path, f"{name}.pickle"), 'wb') as output_file:
            pickle.dump(features_df, output_file, pickle.HIGHEST_PROTOCOL)

# the output is very long, I prefer to clear it after each run.
