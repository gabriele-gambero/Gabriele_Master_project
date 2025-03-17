# %% [markdown]
# # Feature extraction via UNI2-h
# 
# Token:
# `MY_TOKEN`
# 
# Link to GitHub repository: https://github.com/mahmoodlab/UNI/tree/main
# 
# Link to HuggingFace page: https://huggingface.co/MahmoodLab/UNI2-h
# 
# 

# %% [markdown]
# # 0.- Setting things up

# %% [markdown]
# When importing all these packages for the first time (only for the first time your run this cell in the current Jupyter session), you'll get some warning messages. Ignore them as they are related to the CPU and are also the product of a bug and compatibility with `Keras` package.

# %% [markdown]
# Setting the environment to work properly is not easy at all. In the main features extraction folder you can find the requirements for the used environment: `example_kimianet_11_8`.\
# Despite the unusual name, it works fine and it created in the following way:\
# ```sh
# conda create -n example_kimianet_11_8 python=3.9 cudnn=9.3.0 cudatoolkit=11.8 -c nvidia -c conda-forge
# ```
# You can change the `cudnn` version to a most recent one, but the most import thing is to have it at least in `9.3.0` as the KimiaNet NN architecture was build in this version.\
# 
# 
# Once you activate the environment, the first installation that you have to do is the Tensorflow one, thanks to this command:\
# ```sh
# python3 -m pip install 'tensorflow[and-cuda]'
# ```

# %%
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

login(token="MY_TOKEN")  # login with your User Access Token, found at https://huggingface.co/settings/tokens


# %%
# pretrained=True needed to load UNI2-h weights (and download weights for the first time)
# timm_kwargs = {
#             'img_size': 224, 
#             'patch_size': 14, 
#             'depth': 24,
#             'num_heads': 24,
#             'init_values': 1e-5, 
#             'embed_dim': 1536,
#             'mlp_ratio': 2.66667*2,
#             'num_classes': 0, 
#             'no_embed_class': True,
#             'mlp_layer': timm.layers.SwiGLUPacked, 
#             'act_layer': torch.nn.SiLU, 
#             'reg_tokens': 8, 
#             'dynamic_img_size': True
#         }
# model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
# transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
# model.eval()


# %%
import os
import subprocess
import regex
import matplotlib.pyplot as plt
#import seaborn as sns
import glob, pickle, pathlib
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

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
model = "UNI2-h"

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
complete_dict_summaries

# %% [markdown]
# Specify the normalisation methods that have been chosen and the `ORIGINAL WSI` with the exact name in the summaries.

# %%
SELECTED_METHODS = ["ORIGINAL WSI", "fromWSI_histomicsTK_macenko_nomasking", "histomicsTK_macenko_nomasking", "stainNET"]

# %% [markdown]
# Setting the output path.

# %%
comparison_of_analyses = f"{SAMPLE_SATAC}_{IMAGE_VERSION_AND_COORDINATE_SYSTEM_SATAC}_&_{SAMPLE_VISIUM}_{IMAGE_VERSION_AND_COORDINATE_SYSTEM_VISIUM}"
features_output_path = f"output/{model}/{comparison_of_analyses}/"

os.makedirs(features_output_path, exist_ok = True)
features_output_path

# %% [markdown]
# # 2. - Preprocessing and Feature Extraction Functions

# %%
summary_satac_100um = pd.read_csv(SUMMARY_SATAC_100um, index_col= [0])
summary_satac_100um

# %% [markdown]
# Let's import the summary files with the paths to the folders and select the paths to the tiles folders for the ORIGINAL WSI and the methods that worked out.

# %%
filter_summaries_by_method = {name: df[df['Normalisation Method'].isin(SELECTED_METHODS)] for name, df in complete_dict_summaries.items()}

# %% [markdown]
# Selecting by the specified normalisation methods and the target.

# %%
filter_summaries_by_method = {name: df[df['Normalisation Method'].isin(SELECTED_METHODS) & df["Target"].isin(["-", TARGET_IS])].reset_index() for name, df in complete_dict_summaries.items()}
filter_summaries_by_method

# %% [markdown]
# Let's perform the real features extraction step.
# 
# Let's set the batch size (accordingly to GPU computational power), the image file format (tipically `.jpg`).\
# The real extraction function will be called inside the dictionary where each key will contain all the info regarding the images in the folder (sample origin, size in micrometers and the eventual normalisation method) and the relative value will be the dataframe originated from the function which will contain all the extracted features per image (images indeces have their own real name).
# 
# In the end, all the feature dataframes will be saved as `.pickle` files in the `./output` folder.

# %%
# ------------------------------------------------------------------------------------------------------
def image_adapter_for_uni2h(path_to_img, transformer_uni2h):
    
    img = Image.open(path_to_img)
    resized_img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_tensor = transformer_uni2h(resized_img)
    correct_tensor = torch.unsqueeze(img_tensor, dim=0) # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)

    return correct_tensor

# ------------------------------------------------------------------------------------------------------
# Function to extract features and run on GPU
def extract_features_uni2h(row, img_format, transformer_to_be_passed, model_to_be_used):
    patch_dir = row["Path to folder"]
    patch_adr_list = [pathlib.Path(x) for x in glob.glob(os.path.join(patch_dir, f'*.{img_format}'))]

    print(f"Number of files found: {len(patch_adr_list)}")
    if len(patch_adr_list) == 0:
        print("No files found. Please check the patch directory and image format.")
        return

    feature_dict = {}

    for img_path in patch_adr_list:
        image_tensor = image_adapter_for_uni2h(img_path, transformer_to_be_passed).to(device)  # Move image tensor to GPU

        with torch.no_grad():  # Disable gradient computation to save memory
            feature_emb = model_to_be_used(image_tensor)  # Run model on GPU
            #features_array = feature_emb.cpu().numpy()  
            features_array = feature_emb.cpu().detach().numpy().squeeze(0) # Move result back to CPU an squeeze to [1536]

        feature_dict[str(os.path.basename(img_path))] = features_array

    print("Feature extraction completed.")
    return pd.DataFrame.from_dict(feature_dict)
# ------------------------------------------------------------------------------------------------------


# pretrained=True needed to load UNI2-h weights (and download weights for the first time)
timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
uni2h_transformer = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the GPU
model = model.to(device)

# Enable cuDNN optimization
torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------------------------------------------

# Defoning params for calling function
all_extracted_features_from_selected_summaries = {} # this dictionary will contains all the DataFrames with all the extracted features
img_format = 'jpg'

# Create a list of (name, row) pairs
entries = [(name, row) for name, df in filter_summaries_by_method.items() for _, row in df.iterrows()]

# Iterate over the list
for name, row in entries:
    key = f"{name}_{row['Normalisation Method']}"  # Using dictionary-like access
    try:
        features = extract_features_uni2h(row, img_format, uni2h_transformer, model)
        all_extracted_features_from_selected_summaries[key] = features # features as dataframes
    except Exception as e:
        print(f"Error processing {key}: {e}")

# ------------------------------------------------------------------------------------------------------

# saving all the dataframes as .pickles files with the specific name from the folder and the normalisation method that was used.
for name, features_df in all_extracted_features_from_selected_summaries.items():
    with open(os.path.join(features_output_path, f"{name}.pickle"), 'wb') as output_file:
        pickle.dump(features_df, output_file, pickle.HIGHEST_PROTOCOL)
        
# Free up cache if needed
torch.cuda.empty_cache()


# %%
# def extract_features_uni2h_with_batch(row, img_format, transformer, model, batch_size=16):
#     patch_dir = row["Path to folder"]
#     patch_adr_list = [pathlib.Path(x) for x in glob.glob(os.path.join(patch_dir, f'*.{img_format}'))]

#     if not patch_adr_list:
#         print("No files found. Please check the patch directory and image format.")
#         return

#     feature_dict = {}
#     batch_images, image_names = [], []

#     for img_path in patch_adr_list:
#         img_tensor = image_adapter_for_uni2h(img_path, transformer)
#         batch_images.append(img_tensor)
#         image_names.append(str(os.path.basename(img_path)))

#         if len(batch_images) == batch_size:
#             batch_tensor = torch.cat(batch_images, dim=0)  # Create a batch
#             with torch.inference_mode():
#                 batch_features = model(batch_tensor)  # Batch inference
#             for name, feature in zip(image_names, batch_features):
#                 feature_dict[name] = feature.numpy()
#             batch_images, image_names = [], []  # Reset batch

#     # Process remaining images if they don’t fit in the last batch
#     if batch_images:
#         batch_tensor = torch.cat(batch_images, dim=0)
#         with torch.inference_mode():
#             batch_features = model(batch_tensor)
#         for name, feature in zip(image_names, batch_features):
#             feature_dict[name] = feature.numpy()

#     return pd.DataFrame.from_dict(feature_dict)


# %%


# all_extracted_features_from_selected_summaries = {}

# # Create a list of (name, row) pairs
# entries = [
#     (name, row) for name, df in filter_summaries_by_method.items() for _, row in df.iterrows()
# ]

# # Iterate over the list
# for name, row in entries:
#     key = f"{name}_{row['Normalisation Method']}"  # Using dictionary-like access
#     try:
#         features = extract_features(row, NETWORK_WEIGHTS_ADDRESS, batch_size, img_format)
#         all_extracted_features_from_selected_summaries[key] = features # features as dataframes
#     except Exception as e:
#         print(f"Error processing {key}: {e}")

# # the gpu_timer warning at the end of the output doesn't affect it. It's a bug related to the versions of some packages like nightly and CUDNN:
# # https://github.com/tensorflow/tensorflow/issues/71791

# # parameters_folder = f"batchsize{batch_size}" if not network_input_patch_width else f"batchsize{batch_size}_inputpathwidth{network_input_patch_width}"  # might be useless

# # saving all the dataframes as .pickles files with the specific name from the folder and the normalisation method that was used.
# for name, features_df in all_extracted_features_from_selected_summaries.items():
#     with open(os.path.join(features_output_path, f"{name}.pickle"), 'wb') as output_file:
#         pickle.dump(features_df, output_file, pickle.HIGHEST_PROTOCOL)
    
# # the output is very long, I prefer to clear it after each run.

# %% [markdown]
# Saving the dataframes as `.pickle` files (this was done in previous cell as well, here is just for comodity and eventual future modifications of the code).

# %%
# for name, features_df in all_extracted_features_from_selected_summaries.items():
#     with open(os.path.join(features_output_path, f"{name}.pickle"), 'wb') as output_file:
#         pickle.dump(features_df, output_file, pickle.HIGHEST_PROTOCOL)

# %% [markdown]
# Let's visualise one of them.

# %%
# all_extracted_features_from_selected_summaries["satac_C1_&_v3_allspots_&_target_is_reference_full_68um_fromWSI_histomicsTK_macenko_nomasking"]

# %% [markdown]
# ---
# **Saving the environment requirements**

# %%
# Save package versions to a .txt file
# with open("example_kimianet_11_8.txt", "w") as f:
#     subprocess.run(["conda", "list", "--explicit"], stdout=f)

# %% [markdown]
# ---

# %% [markdown]
# # Final - Ending the working session on the GPU

# %% [markdown]
# This is what ChatGPT propose me, but it doesn't work...

# %%
# from tensorflow.keras import backend as K
# #import gc

# # Clear Keras session
# K.clear_session()
# # Force garbage collection
# #gc.collect()


# %% [markdown]
# A way more direct approach could be killing all the processes:
# 
# ```{bash}
# pkill -u <user_name>
# ```

# %%
# from numba import cuda

# device = cuda.get_current_device()
# device
# device.reset()

# %%
# print(device)

# %% [markdown]
# # discarded codes

# %%
# patch_dir = f"../1_tiling/outputs/{sample}/tiling_output/"
# extracted_features_save_adr = f"./models/{model}/output/{sample}/extracted_features_{model}_{sample}.pickle"
# #tissuetypefile_path = "../data/data_for_34C/V10F03-034_C_S7_Wenwen-annotations.csv" # pathologist annotation

# # Extract the directory path only (exclude filename)
# directory_path = os.path.dirname(extracted_features_save_adr)

# # Check if the path exists, if not, create the directories
# for path in [directory_path]:
#     if not os.path.exists(path):
#         # Create the directory and any necessary intermediate directories  
#         os.makedirs(path)
#         print(f"Created directory: {path}")  
#     else:  
#         print(f"Directory already exists: {path}")

# %%
# def preprocessing_fn(input_batch, network_input_patch_width):
#     org_input_size = tf.shape(input_batch)[1]
#     scaled_input_batch = tf.cast(input_batch, 'float32') / 255.0  # Ensuring dtype is float32
#     resized_input_batch = tf.cond(tf.equal(org_input_size, network_input_patch_width),
#                                   lambda: scaled_input_batch, 
#                                   lambda: tf.image.resize(scaled_input_batch, 
#                                                           (network_input_patch_width, network_input_patch_width)))
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     data_format = "channels_last"
#     mean_tensor = tf.constant(-np.array(mean), dtype=tf.float32)  # Ensuring mean tensor is float32
#     standardized_input_batch = tf.keras.backend.bias_add(resized_input_batch, mean_tensor, data_format)
#     standardized_input_batch /= std
#     return standardized_input_batch


# # ------------------------------------------------------------------------------------------------------
# def kimianet_feature_extractor(network_input_patch_width, weights_address):
#     if not os.path.exists(weights_address):
#         raise FileNotFoundError(f"Weights file not found at {weights_address}")
    
#     dnx = DenseNet121(include_top=False, weights=weights_address, 
#                       input_shape=(network_input_patch_width, network_input_patch_width, 3), pooling='avg')
#     kn_feature_extractor = Model(inputs=dnx.input, outputs=GlobalAveragePooling2D()(dnx.layers[-3].output))
#     kn_feature_extractor_seq = Sequential([Lambda(preprocessing_fn, 
#                                                   arguments={'network_input_patch_width': network_input_patch_width}, 
#                                                   input_shape=(None, None, 3), dtype=tf.uint8)])
#     kn_feature_extractor_seq.add(kn_feature_extractor)
#     return kn_feature_extractor_seq


# # ------------------------------------------------------------------------------------------------------
# def extract_features(patch_dir, network_weights_address, 
#                      network_input_patch_width, batch_size, img_format):
#     feature_extractor = kimianet_feature_extractor(network_input_patch_width, network_weights_address)
#     patch_adr_list = [pathlib.Path(x) for x in glob.glob(os.path.join(patch_dir, f'*.{img_format}'))]
    
#     # Debug: Print number of files found
#     print(f"Number of files found: {len(patch_adr_list)}")
    
#     if len(patch_adr_list) == 0:
#         print("No files found. Please check the patch directory and image format.")
#         return
    
#     feature_dict = {}

#     for batch_st_ind in tqdm(range(0, len(patch_adr_list), batch_size)):
#         batch_end_ind = min(batch_st_ind + batch_size, len(patch_adr_list))
#         batch_patch_adr_list = patch_adr_list[batch_st_ind:batch_end_ind]
        
#         # Debug: Print current batch size
#         print(f"Processing batch from index {batch_st_ind} to {batch_end_ind}")
        
#         patch_batch = np.array([skimage.io.imread(str(x)) for x in batch_patch_adr_list])
#         batch_features = feature_extractor.predict(patch_batch)
#         feature_dict.update(dict(zip([x.stem for x in batch_patch_adr_list], list(batch_features))))
        
#         # with open(extracted_features_save_adr, 'wb') as output_file:
#         #     pickle.dump(feature_dict, output_file, pickle.HIGHEST_PROTOCOL)
    
#     print("Feature extraction completed.")
#     return pd.DataFrame.from_dict(feature_dict)


# %% [markdown]
# **With patch width**

# %%
# network_input_patch_width = 224 #1000
# batch_size = 32 #16
# img_format = 'jpg'
# #use_gpu = True

# all_extracted_features_from_selected_summaries = {}

# # Create a list of (name, row) pairs
# entries = [
#     (name, row) for name, df in filter_summaries_by_method.items() for _, row in df.iterrows()
# ]

# # Iterate over the list
# for name, row in entries:
#     key = f"{name}_{row['Normalisation Method']}"  # Using dictionary-like access
#     try:
#         features = extract_features(row, NETWORK_WEIGHTS_ADDRESS, network_input_patch_width, batch_size, img_format)
#         all_extracted_features_from_selected_summaries[key] = features
#     except Exception as e:
#         print(f"Error processing {key}: {e}")



