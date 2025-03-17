# Be careful, the code takes some time to run and it runs on the CPU.

import os
import subprocess
import torchvahadane
from torchvahadane import TorchVahadaneNormalizer
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
import datetime

starttime = datetime.datetime.now()



# setting the working directory
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/2_image_normalisation/")
print(os.getcwd())

# setting a single GPU as the only visible one
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 0 = first GPU, 1 = second GPU

# ---------------------------------------------------------------------------------
# setting the paths
normalisation_method = 'torch_vahadane_cpu'

#INPUT_FOLDER = "../1_tiling/output/satac_C1/tiling_output/v3_allspots/tiles_100/"  # Replace with the path to your folder containing images
# INPUT_FOLDER = "../1_tiling/output/visium_2022_FF_WG_10X/tiling_output/img_not_changed_allspots/tiles_100"  # Replace with the path to your folder containing images
INPUT_FOLDER = "../1_tiling/output/visium_FFPE_dcis_idc_10X/tiling_output/img_not_changed_allspots/tiles_100"  # Replace with the path to your folder containing images

tiles_info = INPUT_FOLDER.split('/')

TARGET_IMAGE_PATH = "reference_images/reference_full.jpeg"
target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[1].split(".")[0]
output_folder = f"./output/{tiles_info[3]}/{tiles_info[5]}/{tiles_info[6]}/{normalisation_method}/{target_temp_path}"
print(output_folder)

# Let's create the output folder files
os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------------------------------------------------
# Load and preprocess the target image
target = Image.open(TARGET_IMAGE_PATH).convert("RGB")
target_array = np.array(target)  # Convert to NumPy array (for OpenCV compatibility)

# Initialize the normalizer
normalizer = TorchVahadaneNormalizer(device='cuda', staintools_estimate=True)
#normalizer.stain_extractor.luminosity_threshold = 0.9  # Increase mask sensitivity
normalizer.fit(target_array)  # Pass NumPy array for compatibility

# ---------------------------------------------------------------------------------
# File to log images that fail normalization
normalisation_fails_file = f"{output_folder}/0_failed_to_normalise.txt" # 0 just for having the file listed as first

with open(normalisation_fails_file, "w") as file:
    file.write("The following are the tiles not normalised:\n")
    
    # Process each image in the input folder
    for filename in os.listdir(INPUT_FOLDER):
        image_path = os.path.join(INPUT_FOLDER, filename)

        # Load and preprocess the image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)  # Convert to NumPy array (for OpenCV compatibility)


        try:
            # Perform normalization
            img_normed_tensor = normalizer.transform(img_array, return_mask=False)
            
            # Convert the normalized tensor to a NumPy array
            img_normed_array = img_normed_tensor.cpu().numpy().clip(0, 255).astype(np.uint8)

            # Convert the normalized image back to PIL format
            img_normed_pil = Image.fromarray(img_normed_array)

            # Save the normalized image
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{normalisation_method}.jpg") # or .png (but it's way bigger)
            img_normed_pil.save(output_path)

            #print(f"Normalized image saved to: {output_path}")
            
        except Exception as e:
            file.write(f"{filename}\n")
            print(f"Error processing {filename}: {e}")


difference =  datetime.datetime.now() - starttime

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

print(f"Finished! The normalisation took {difference} seconds!")
