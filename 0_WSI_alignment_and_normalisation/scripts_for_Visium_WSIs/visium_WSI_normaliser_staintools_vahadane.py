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

# %load_ext autoreload
# %autoreload 2

#%matplotlib inline

starttime = datetime.datetime.now()

os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/0_WSI_alignment_and_normalisation/")
print(os.getcwd())

# SET PATHS, HERE IS THE MOST IMPORTANT STEP, BE CAREFUL WITH IT.
INPUT_WSI = "../data/spatial_atac/modified_images/BCSA4_A2_sATAC_C1_adjacent-Spot000001_v3_newrot_newcrop_realcolors_nofakescaling.jpg"  # Replace with the path to your folder containing the WSI
wsi_info = INPUT_WSI.split('/')[-1].split("_")


# Define the paths
TARGET_IMAGE_PATH = "../2_image_normalisation/reference_images/reference_full.jpeg"
target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[-1].split(".")[0]
normalisation_method = 'staintools_vahadane'
output_folder = f"./output/{wsi_info[2]}_{wsi_info[3]}/wsi_{wsi_info[5]}/{normalisation_method}/{target_temp_path}"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# File to log normalization failures
normalisation_fails_file = f"{output_folder}/0_failed_to_normalise.txt"

# Compute target statistics from target image
target = staintools.read_image(TARGET_IMAGE_PATH)
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(target)

def process_image(image_path, output_path):
    try:
        # Read and normalize the image
        img = Image.open(image_path).convert("RGB")
        wsi_array = np.array(img)

        # Define crop region
        top_left = (3500, 3300)
        bottom_right = (wsi_array.shape[1] - 5200, wsi_array.shape[0] - 4000)  # (width, height)

        # Crop the region
        crop = wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]        
        tissue_rgb_normalized = normalizer.transform(crop)

        # pasting back the normalised image as an array
        wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = tissue_rgb_normalized
        
        # Convert to PIL and save
        img_normed_pil = Image.fromarray((wsi_array).astype('uint8'))
        output_image_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_st_vahadane_normalized.jpg")
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