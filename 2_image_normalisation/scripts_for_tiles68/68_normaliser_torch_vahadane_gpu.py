
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
from torch.backends import cudnn

# GPU optimization
cudnn.benchmark = True

starttime = datetime.datetime.now()

# setting the working directory
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/2_image_normalisation/")
print(os.getcwd())

# setting a single GPU as the only visible one
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 0 = first GPU, 1 = second GPU

# ---------------------------------------------------------------------------------
# setting the paths
normalisation_method = 'torch_vahadane_gpu'

#INPUT_FOLDER = "../1_tiling/output/satac_C1/tiling_output/v3_allspots/tiles_68/"  # Replace with the path to your folder containing images
#INPUT_FOLDER = "../1_tiling/output/visium_2022_FF_WG_10X/tiling_output/img_not_changed_allspots/tiles_68"  # Replace with the path to your folder containing images
INPUT_FOLDER = "../1_tiling/output/visium_FFPE_dcis_idc_10X/tiling_output/img_not_changed_allspots/tiles_68"  # Replace with the path to your folder containing images

tiles_info = INPUT_FOLDER.split('/')

TARGET_IMAGE_PATH = "reference_images/reference_full.jpeg"
target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[1].split(".")[0]
output_folder = f"./output/{tiles_info[3]}/{tiles_info[5]}/{tiles_info[6]}/{normalisation_method}/{target_temp_path}"
print(output_folder)

# Let's create the output folder files
os.makedirs(output_folder, exist_ok=True)


# ---------------------------------------------------------------------------------
gpu = "cuda"
# Load and convert to NumPy array (for OpenCV compatibility)
target = Image.open(TARGET_IMAGE_PATH).convert("RGB")
target = torch.Tensor(np.array(target))  # Convert to PyTorch tensor

# Transpose dimensions to match Kornia's expected format
target = target.permute(2, 0, 1).to(gpu)

# Initialize the normalizer
normalizer = TorchVahadaneNormalizer(device=gpu, staintools_estimate=False) # setting it on False will make it run with GPU acceleration
#normalizer.stain_extractor.luminosity_threshold = 0.9 # increasing the mask sensitivity. Normally is 0.8
normalizer.fit(target)  # Pass NumPy array for compatibility

normalisation_fails_file = f"{output_folder}/0_failed_to_normalise.txt" # 0 just for having the file listed as first

# Process each image in the input folder
with open(normalisation_fails_file, "w") as file:
    file.write("The following tiles haven't normalised:\n")

    for filename in os.listdir(INPUT_FOLDER):
        image_path = os.path.join(INPUT_FOLDER, filename)
        
        # Load and preprocess the image
        img = Image.open(image_path).convert("RGB")
        img = torch.Tensor(np.array(img))  # Convert to PyTorch tensor

        # Transpose dimensions to match Kornia's expected format
        img = img.permute(2, 0, 1).to(gpu)#.unsqueeze(0)
        
        try:
            # Perform normalization
            img_normed_tensor = normalizer.transform(img)
            
            # Convert the normalized image back to PIL format
            img_normed_pil = torchvision.transforms.functional.to_pil_image(img_normed_tensor)

            # Save the normalized image
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{normalisation_method}.jpg")
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
