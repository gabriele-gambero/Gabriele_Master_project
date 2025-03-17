import histomicstk
import os
import argparse
import datetime
import numpy as np
from PIL import Image
from matplotlib import pylab as plt
from skimage.transform import resize
from histomicstk.saliency.tissue_detection import get_tissue_mask
from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import deconvolution_based_normalization

# Parse command line arguments
parser = argparse.ArgumentParser(description='Normalize a WSI using Macenko method with masking')
parser.add_argument('input_wsi', type=str, help='Path to the input WSI image')
parser.add_argument('target_image', type=str, help='Path to the target image for normalization')
args = parser.parse_args()

starttime = datetime.datetime.now()

# setting the working directory
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/0_WSI_alignment_and_normalisation/")
print(os.getcwd())

# ---------------------------------------------------------------------------------
# setting the paths
normalisation_method = 'histomicsTK_macenko_withmasking'

INPUT_WSI = args.input_wsi
TARGET_IMAGE_PATH = args.target_image

# Dictionary for all the Visium approaches
wsis_dict = {
    "../data/visium_2022_FF_WG/input_files/Visium_Human_Breast_Cancer_image.tif": {
        "SAMPLE": "2022_FF_WG_10X",
        "IMAGE_VERSION": "img_not_changed",
        "CROPPING_COORDINATES": [(11520, 4176), (29760, 23472)]
    },
    '../data/visium_ffpe/input_files/Visium_FFPE_Human_Breast_Cancer_image.tif': {
        "SAMPLE": "FFPE_dcis_idc_10X",
        "IMAGE_VERSION": "img_not_changed",
        "CROPPING_COORDINATES": [(5535, 4770), (22320, 21510)]
    }
}

if INPUT_WSI in wsis_dict:
    selected_data = wsis_dict[INPUT_WSI]
    selected_sample = selected_data["SAMPLE"]
    selected_image_version = selected_data["IMAGE_VERSION"]
    selected_cropping_coordinates = selected_data["CROPPING_COORDINATES"]
else:
    print("Error: Image path not found in dataset.")
    exit(1)

print(f"Sample name: {selected_sample}\nImage version: {selected_image_version}\nCropping between: {selected_cropping_coordinates[0]} and {selected_cropping_coordinates[1]}")

# Define the paths
target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[-1].split(".")[0]
output_folder = f"./output/{selected_sample}/wsi_{selected_image_version}/{normalisation_method}/{target_temp_path}"

# Let's create the output folder files
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------------
# Compute target statistics from target image
target = Image.open(TARGET_IMAGE_PATH).convert("RGB")
im_target = np.array(target)
#I_0 = 240 # 240 is the default for brightfield images. Otherwise 255.
#W_target= rgb_separate_stains_macenko_pca(target_array, I_0)
#print(W_target)
#W_target = None


# File to log normalization failures
normalisation_fails_file = f"{output_folder}/0_failed_to_normalise_macenko.txt"

with open(normalisation_fails_file, "w") as file:
    file.write("The following tiles have failed normalization:\n")
    
    filename = os.path.splitext(os.path.basename(INPUT_WSI))[0]

    try:
        # Load image
        img = Image.open(INPUT_WSI).convert("RGB")
        wsi_array = np.array(img)

        # Define crop region
        top_left = selected_cropping_coordinates[0] # (width, height)
        bottom_right = selected_cropping_coordinates[1]  

        # Crop the region
        crop = wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        # Mask extraction 
        # mask_out, _ = get_tissue_mask(crop, deconvolve_first=False,
        #     n_thresholding_steps=1, sigma=3.0, min_size=5)
        # mask_out, _ = get_tissue_mask(crop, deconvolve_first=False,
        #     n_thresholding_steps=2, sigma=0.01, min_size=20)
        mask_out, _ = get_tissue_mask(crop, deconvolve_first=False,
            n_thresholding_steps=2, sigma=0.5, min_size=10)
        mask_out = resize(mask_out == 0, output_shape=(crop.shape[0] // 2, crop.shape[1] // 2), 
            order=0, preserve_range=True) == 1
        mask_out = resize(mask_out, output_shape=crop.shape[:2], order=0, preserve_range=True)
        # mask_out = resize(mask_out == 0, output_shape=crop.shape[:2],
        #     order=0, preserve_range=True) == 1
        
        # Perform Macenko normalization
        tissue_rgb_normalized = deconvolution_based_normalization(
            crop,
            im_target=im_target,
            stain_unmixing_routine_params={'stains': ['hematoxylin', 'eosin'], 'stain_unmixing_method': 'macenko_pca'},
            mask_out = mask_out
        )
        
        # pasting back the normalised image as an array
        wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = tissue_rgb_normalized
        
        # Convert to PIL and save
        img_normed_pil = Image.fromarray(wsi_array.astype('uint8'))
        output_path = os.path.join(output_folder, f"{filename}_{normalisation_method}.jpg")
        img_normed_pil.save(output_path)
        
        # Convert the mask to PIL and save
        mask_pil = Image.fromarray((mask_out*255).astype('uint8'))
        output_path_mask = os.path.join(output_folder, "mask_macenko_wsi.jpg")
        mask_pil.save(output_path_mask)
        
    except Exception as e:
        file.write(f"{filename}: {str(e)}\n")
        print(f"Error processing {filename}: {e}")

difference = datetime.datetime.now() - starttime

# Delete previous time log files
for filename in os.listdir(output_folder):
    if filename.startswith("0_started_"):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

# Save the start and finish time in the file's name
with open(f"{output_folder}/0_started_at_{starttime}_finished_at_{datetime.datetime.now()}.txt", "w") as file:
    file.write(f"The run started at {starttime} and finished at {datetime.datetime.now()}.")

print(f"Finished! The normalisation took {difference} seconds!")



