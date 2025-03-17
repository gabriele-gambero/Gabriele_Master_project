# Be careful, the code takes some time to run and it runs on the CPU.

import os
import subprocess
import torchvahadane
from torchvahadane import TorchVahadaneNormalizer
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely, or set to a large value like 3_000_000_000
import numpy as np
import torch
torch.cuda.empty_cache()
import torchvision
import torchvision.transforms.functional
import datetime

starttime = datetime.datetime.now()



# setting the working directory
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/0_WSI_alignment_and_normalisation/")
print(os.getcwd())

# setting a single GPU as the only visible one
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 = first GPU, 1 = second GPU

# ---------------------------------------------------------------------------------
# setting the paths
normalisation_method = 'torch_vahadane_cpu'

INPUT_WSI = "../data/spatial_atac/modified_images/BCSA4_A2_sATAC_C1_adjacent-Spot000001_v3_newrot_newcrop_realcolors_nofakescaling.jpg"  # Replace with the path to your folder containing the WSI
wsi_info = INPUT_WSI.split('/')[-1].split("_")

# Define the paths
TARGET_IMAGE_PATH = "../2_image_normalisation/reference_images/reference_full.jpeg"
target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[-1].split(".")[0]
output_folder = f"./output/{wsi_info[2]}_{wsi_info[3]}/wsi_{wsi_info[5]}/{normalisation_method}/{target_temp_path}"

# Let's create the output folder files
os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------------------------------------------------
# Load and preprocess the target image
target = Image.open(TARGET_IMAGE_PATH).convert("RGB")
target_array = np.array(target)  # Convert to NumPy array (for OpenCV compatibility)

# Initialize the normalizer
normalizer = TorchVahadaneNormalizer(device='cpu', staintools_estimate=True)
#normalizer.stain_extractor.luminosity_threshold = 0.9  # Increase mask sensitivity
normalizer.fit(target_array)  # Pass NumPy array for compatibility

# ---------------------------------------------------------------------------------
# File to log images that fail normalization
normalisation_fails_file = f"{output_folder}/0_failed_to_normalise.txt" # 0 just for having the file listed as first

# Process the WSI
with open(normalisation_fails_file, "w") as file:
    file.write("The following tiles haven't normalised:\n")

    filename = os.path.splitext(os.path.basename(INPUT_WSI))[0]
    
    # Load and preprocess the image
    img = Image.open(INPUT_WSI).convert("RGB")
    img_array = np.array(img)  # Convert to NumPy array (for OpenCV compatibility)
    
    try:
        # Perform normalization
        img_normed_tensor = normalizer.transform(img_array, return_mask=False)
        
        # Convert the normalized tensor to a NumPy array
        img_normed_array = img_normed_tensor.cpu().numpy().clip(0, 255).astype(np.uint8)

        # Convert the normalized image back to PIL format
        img_normed_pil = Image.fromarray(img_normed_array)

        # Save the normalized image
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{normalisation_method}_cpu.jpg")
        img_normed_pil.save(output_path)

        print(f"Normalized image saved to: {output_path}")
    
    except Exception as e:
        file.write(f"{filename}\n")
        print(f"Error processing {filename}: {e}")


# eventually deleting the previous time log file
for filename in os.listdir(output_folder):
    if filename.startswith("0_started_"):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):  # Check if it is a file
            os.remove(file_path)      # Delete the file
            print(f"Deleted: {file_path}")

difference =  datetime.datetime.now() - starttime

with open(f"{output_folder}/0_started_at_{starttime}_finished_at_{datetime.datetime.now()}.txt", "w") as file:
    file.write(f"The run started at {starttime} and finished at {datetime.datetime.now()}.")

print(f"Finished! The normalisation took {difference} seconds!")