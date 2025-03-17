# The script must be launched from the `he_stainnet` environment

import os
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/0_WSI_alignment_and_normalisation/")
print(os.getcwd())
# setting a single GPU as the only visible one
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 0 = first GPU, 1 = second GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'

import numpy as np
import subprocess
import staintools
import torch
torch.cuda.empty_cache()
import sys
import datetime
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely, or set to a large value like 3_000_000_000
import matplotlib.pyplot as plt
import datetime

#%matplotlib inline

sys.path.append('../data/packages/StainNet/')

from models import StainNet, ResnetGenerator


pretrained_models_path = '../data/packages/StainNet/checkpoints/camelyon16_dataset/'
print(os.listdir(pretrained_models_path))

def norm(image):
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = ((image / 255) - 0.5) / 0.5
    image=image[np.newaxis, ...]
    image=torch.from_numpy(image)
    return image

def un_norm(image):
    image = image.cpu().detach().numpy()[0]
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose((1,2,0))
    return image


# ---------------------------------------------------------------------------------
# setting the paths
INPUT_WSI = "../data/spatial_atac/modified_images/BCSA4_A2_sATAC_C1_adjacent-Spot000001_v3_newrot_newcrop_realcolors_nofakescaling.jpg"  # Replace with the path to your folder containing the WSI
wsi_info = INPUT_WSI.split('/')[-1].split("_")

# Define the paths
normalisation_method = 'stainGAN'
output_folder_staingan = f"./output/{wsi_info[2]}_{wsi_info[3]}/wsi_{wsi_info[5]}/{normalisation_method}"

# Path for model B
output_folder_model_B = os.path.join(output_folder_staingan, "model_B")
# Let's create the output folder files
os.makedirs(output_folder_model_B, exist_ok=True)

starttime = datetime.datetime.now()

# ---------------------------------------------------------------------------------
# load pretrained StainGAN
model_GAN = ResnetGenerator(3, 3, ngf=64, norm_layer=torch.nn.InstanceNorm2d, n_blocks=9).cuda()
model_GAN.load_state_dict(torch.load('../data/packages/StainNet/checkpoints/camelyon16_dataset/latest_net_G_B.pth'))

# ---------------------------------------------------------------------------------
starttime = datetime.datetime.now()

# File to log images that fail normalization
normalisation_fails_file = f"{output_folder_model_B}/0_failed_to_normalise.txt" # 0 just for having the file listed as first

with open(normalisation_fails_file, "w") as file:
    file.write("The following are the tiles not normalised:\n")
    
    filename = os.path.basename(INPUT_WSI)
    
    # Load and preprocess the image
    img = Image.open(INPUT_WSI).convert("RGB")
    wsi_array = np.array(img)

    # Define crop region
    top_left = (3500, 3300)
    bottom_right = (wsi_array.shape[1] - 5200, wsi_array.shape[0] - 4000)  # (width, height)

    # Crop the region
    crop = wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    
    try:
        # Perform normalization
        model_GAN.eval()
        with torch.no_grad():
            img_gan=model_GAN(norm(crop).cuda())
            img_normed_array=un_norm(img_gan)

        # pasting back the normalised image as an array
        wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = img_normed_array
        
        # Convert the normalized image back to PIL format
        img_normed_pil = Image.fromarray(wsi_array)

        # Save the normalized image
        output_path = os.path.join(output_folder_model_B, f"{os.path.splitext(filename)[0]}_{normalisation_method}_StainGAN_modelB.jpg") # or .png (but it's way bigger)
        img_normed_pil.save(output_path)

        #print(f"Normalized image saved to: {output_path}")
        
    except Exception as e:
        file.write(f"{filename}\n")
        print(f"Error processing {filename}: {e}")


difference =  datetime.datetime.now() - starttime

# eventually deleting the previous time log file
for filename in os.listdir(output_folder_model_B):
    if filename.startswith("0_started_"):
        file_path = os.path.join(output_folder_model_B, filename)
        if os.path.isfile(file_path):  # Check if it is a file
            os.remove(file_path)      # Delete the file
            print(f"Deleted: {file_path}")

# saving the start and finish time in the file's name for simplicity in the reading.
with open(f"{output_folder_model_B}/0_started_at_{starttime}_finished_at_{datetime.datetime.now()}.txt", "w") as file:
    file.write(f"The run started at {starttime} and finished at {datetime.datetime.now()}.")

print(f"Finished! The normalisation took {difference} seconds!")

