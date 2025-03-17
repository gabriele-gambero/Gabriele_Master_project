# must be run in he_staintools environment

from __future__ import division

import os
import datetime

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()  # Disable OpenCV image size limit
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable Pillow image size limit

import cv2
import staintools
import numpy as np
import matplotlib.pyplot as plt
import argparse
# %load_ext autoreload
# %autoreload 2
#%matplotlib inline


# Parse command line arguments
parser = argparse.ArgumentParser(description='Normalize a WSI using Macenko method from Staintools package.')
parser.add_argument('input_wsi', type=str, help='Path to the input WSI image')
parser.add_argument('target_image', type=str, help='Path to the target image for normalization')
args = parser.parse_args()

starttime = datetime.datetime.now()

os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/0_WSI_alignment_and_normalisation/")
print(os.getcwd())

# ---------------------------------------------------------------------------------
# SET PATHS, HERE IS THE MOST IMPORTANT STEP, BE CAREFUL WITH IT.
INPUT_WSI = args.input_wsi
TARGET_IMAGE_PATH = args.target_image
normalisation_method = 'staintools_macenko'

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


target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[-1].split(".")[0]
output_folder = f"./output/{selected_sample}/wsi_{selected_image_version}/{normalisation_method}/{target_temp_path}"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------------
# File to log normalization failures
normalisation_fails_file = f"{output_folder}/0_failed_to_normalise.txt"

# Compute target statistics from target image
target = staintools.read_image(TARGET_IMAGE_PATH)
normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(target)

# ---------------------------------------------------------------------------------
def process_image(image_path, output_path):
    try:
        # Read and normalize the image
        img = Image.open(image_path).convert("RGB")
        wsi_array = np.array(img)

        # Define crop region
        top_left = selected_cropping_coordinates[0]
        bottom_right = selected_cropping_coordinates[1]

        # Crop the region
        crop = wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]        
        tissue_rgb_normalized = normalizer.transform(crop)

        # pasting back the normalised image as an array
        wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = tissue_rgb_normalized
        
        # Convert to PIL and save
        img_normed_pil = Image.fromarray((wsi_array).astype('uint8'))
        output_image_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_st_macenko_normalized.jpg")
        img_normed_pil.save(output_image_path)

    except Exception as e:
        # Log failures
        with open(normalisation_fails_file, "w") as file:
            file.write(f"{image_path}: {str(e)}\n")
            print(e)


if __name__ == "__main__":
    process_image(INPUT_WSI, output_folder)

    # eventually deleting the previous time log file
    for filename in os.listdir(output_folder):
        if filename.startswith("0_started_"):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):  # Check if it is a file
                os.remove(file_path)      # Delete the file
                print(f"Deleted: {file_path}")

    # saving the start and finish time in the file's name for simplicity in the reading.
    with open(f"{output_folder}/0_started_at_{starttime}_finished_at_{datetime.datetime.now()}.txt", "w") as file:
        file.write(f"The run started at {starttime} and finished at {datetime.datetime.now()}.")
