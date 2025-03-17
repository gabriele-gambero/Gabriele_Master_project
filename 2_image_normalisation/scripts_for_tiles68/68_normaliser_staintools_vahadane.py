from __future__ import division

import staintools
#%load_ext autoreload
#%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool, Lock
import os
import datetime

#%matplotlib inline

starttime = datetime.datetime.now()

os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/2_image_normalisation/")
print(os.getcwd())

# SET PATHS, HERE IS THE MOST IMPORTANT STEP, BE CAREFUL WITH IT.
#INPUT_FOLDER = "../1_tiling/output/satac_C1/tiling_output/v3_allspots/tiles_68/"  # Replace with the path to your folder containing images
# INPUT_FOLDER = "../1_tiling/output/visium_2022_FF_WG_10X/tiling_output/img_not_changed_allspots/tiles_68"  # Replace with the path to your folder containing images
INPUT_FOLDER = "../1_tiling/output/visium_FFPE_dcis_idc_10X/tiling_output/img_not_changed_allspots/tiles_68"  # Replace with the path to your folder containing images

tiles_info = INPUT_FOLDER.split('/')



# Define the paths
TARGET_IMAGE_PATH = "reference_images/reference_full.jpeg"
target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[1].split(".")[0]
normalisation_method = 'staintools_vahadane'
output_folder = f"./output/{tiles_info[3]}/{tiles_info[5]}/{tiles_info[6]}/{normalisation_method}/{target_temp_path}"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# File to log normalization failures
normalisation_fails_file = f"{output_folder}/0_failed_to_normalise.txt"
with open(normalisation_fails_file, "w") as file:
    file.write("The following tiles have failed normalization:\n")

# Compute target statistics from target image
target = staintools.read_image(TARGET_IMAGE_PATH)
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(target)

def process_image(image_path, output_path):
    try:
        # Read and normalize the image
        to_transform = staintools.read_image(image_path)
        transformed = normalizer.transform(to_transform)

        # Convert the transformed array back to a PIL image
        img_normed_pil = Image.fromarray(np.uint8(transformed))

        # Define the output file path
        output_image_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_st_vahadane_normalized.jpg")

        # Save the normalized image
        img_normed_pil.save(output_image_path)

    except Exception as e:
        # Log failures
        with open(normalisation_fails_file, "a") as file:
            file.write(f"{image_path}: {str(e)}\n")

def normalize_images_sequentially(input_folder, output_path):
    image_paths = [os.path.join(input_folder, image) for image in os.listdir(input_folder) if image.endswith(('.jpg', '.png', '.jpeg'))]
    
    for image_path in image_paths:
        process_image(image_path, output_path)

if __name__ == "__main__":
    normalize_images_sequentially(INPUT_FOLDER, output_folder)

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
