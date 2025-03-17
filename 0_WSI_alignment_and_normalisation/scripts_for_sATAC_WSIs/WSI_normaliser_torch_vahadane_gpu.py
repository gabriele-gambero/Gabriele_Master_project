# very important: before launching the script you should also run this script in the terminal:
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# this allows PyTorch to allocate memory in smaller chunks or "segments," making the memory usage more efficient and reducing fragmentation.
import os
import subprocess
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely, or set to a large value like 3_000_000_000
import numpy as np
# GPU optimization
import torch
torch.cuda.empty_cache()
import torchvision
import torchvision.transforms.functional
import datetime
from torch.backends import cudnn
cudnn.benchmark = True
import torchvahadane
from torchvahadane import TorchVahadaneNormalizer
from torch.amp import autocast

starttime = datetime.datetime.now()

# setting the working directory
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/0_WSI_alignment_and_normalisation/")
print(os.getcwd())

# setting a single GPU as the only visible one
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 = first GPU, 1 = second GPU

# ---------------------------------------------------------------------------------
# setting the paths
normalisation_method = 'torch_vahadane_gpu'

# SET PATHS, HERE IS THE MOST IMPORTANT STEP, BE CAREFUL WITH IT.
INPUT_WSI = "../data/spatial_atac/modified_images/BCSA4_A2_sATAC_C1_adjacent-Spot000001_v3_newrot_newcrop_realcolors_nofakescaling.jpg"  # Replace with the path to your folder containing the WSI
wsi_info = INPUT_WSI.split('/')[-1].split("_")

# Define the paths
TARGET_IMAGE_PATH = "../2_image_normalisation/reference_images/reference_full.jpeg"
target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[-1].split(".")[0]
output_folder = f"./output/{wsi_info[2]}_{wsi_info[3]}/wsi_{wsi_info[5]}/{normalisation_method}/{target_temp_path}"

# Let's create the output folder files
os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------------------------------------------------
gpu = "cuda"
# Load and convert to NumPy array (for OpenCV compatibility)
target = Image.open(TARGET_IMAGE_PATH).convert("RGB")
target = torch.Tensor(np.array(target))  # Convert to PyTorch tensor

# Transpose dimensions to match Kornia's expected format
target = target.permute(2, 0, 1).to(gpu)

with autocast(device_type='cuda', dtype=torch.float32):
    # Initialize the normalizer
    normalizer = TorchVahadaneNormalizer(device=gpu, staintools_estimate=False) # setting it on False will make it run with GPU acceleration
    #normalizer.stain_extractor.luminosity_threshold = 0.9 # increasing the mask sensitivity. Normally is 0.8
    normalizer.fit(target)  # Pass NumPy array for compatibility

normalisation_fails_file = f"{output_folder}/0_failed_to_normalise.txt" # 0 just for having the file listed as first

with autocast(device_type='cuda', dtype=torch.float16):
    # Process the WSI
    with open(normalisation_fails_file, "w") as file:
        file.write("The following tiles haven't normalised:\n")

        filename = os.path.basename(INPUT_WSI)[0]
        
        # Load and preprocess the image
        img = Image.open(INPUT_WSI).convert("RGB")
        img = torch.Tensor(np.array(img))  # Convert to PyTorch tensor

        # Transpose dimensions to match Kornia's expected format
        img = img.permute(2, 0, 1).to(gpu)#.unsqueeze(0)
        
        try:
            # Perform normalization
            img_normed_tensor = normalizer.transform(img)
            
            # Convert the normalized image back to PIL format
            img_normed_pil = torchvision.transforms.functional.to_pil_image(img_normed_tensor)

            # Save the normalized image
            output_path = os.path.join(output_folder, f"{filename}_{normalisation_method}_gpu.jpg")
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
