import histomicstk
import os
import matplotlib.pyplot as plt
import subprocess
from PIL import Image
import numpy as np
#import girder_client
from skimage.transform import resize
from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
from histomicstk.preprocessing.color_deconvolution.\
    color_deconvolution import color_deconvolution_routine, stain_unmixing_routine
from histomicstk.preprocessing.augmentation.\
    color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration
import datetime
    
starttime = datetime.datetime.now()
    
# setting the working directory
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/0_WSI_alignment_and_normalisation/")
print(os.getcwd())

# ----------------------------------------------------
use_default_target = True   # True if you want to use the default values from function example, False for using the choosen target image 
normalisation_method = 'histomicsTK_reinhard_withmasking'

# SET PATHS, HERE IS THE MOST IMPORTANT STEP, BE CAREFUL WITH IT.
INPUT_WSI = "../data/spatial_atac/modified_images/BCSA4_A2_sATAC_C1_adjacent-Spot000001_v3_newrot_newcrop_realcolors_nofakescaling.jpg"  # Replace with the path to your folder containing the WSI
wsi_info = INPUT_WSI.split('/')[-1].split("_")
# ----------------------------------------------------


if use_default_target:
    # Setting up the paths for default target    
    output_folder = f"./output/{wsi_info[2]}_{wsi_info[3]}/wsi_{wsi_info[5]}/{normalisation_method}/target_is_default"
    print(output_folder)
    
    # Using the target values from the function example
    # color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
    cnorm = {'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
             'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),}
    
else:
    # Setting up the paths using a personalised image as target
    TARGET_IMAGE_PATH = "../2_image_normalisation/reference_images/reference_full.jpeg"
    target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[-1].split(".")[0]
    output_folder = f"./output/{wsi_info[2]}_{wsi_info[3]}/wsi_{wsi_info[5]}/{normalisation_method}/{target_temp_path}"
    print(output_folder)
    
    # Compute target statistics from target image
    target = Image.open(TARGET_IMAGE_PATH).convert("RGB")
    target_array = np.array(target)
    cnorm = {'mu': target_array.mean(axis=(0, 1)),
             'sigma': target_array.std(axis=(0, 1)),}
    

# Let's create the output folder files
os.makedirs(output_folder, exist_ok=True)


# File to log normalization failures
normalisation_fails_file = f"{output_folder}/0_failed_to_normalise.txt"

with open(normalisation_fails_file, "w") as file:
    file.write("The following tiles have failed normalization:\n")
    
    filename = os.path.splitext(os.path.basename(INPUT_WSI))[0]

    try:
        # Load image
        img = Image.open(INPUT_WSI).convert("RGB")
        wsi_array = np.array(img)

        # Define crop region
        top_left = (3500, 3300)
        bottom_right = (wsi_array.shape[1] - 5200, wsi_array.shape[0] - 4000)  # (width, height)

        # Crop the region
        crop = wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        # Mask extraction without resizing because it's not a WSI, it's quite a small image
        mask_out, _ = get_tissue_mask(crop, deconvolve_first=True,
            n_thresholding_steps=1, sigma=1.5, min_size=30)
        mask_out = resize(mask_out == 0, output_shape=crop.shape[:2],
            order=0, preserve_range=True) == 1
        
        # Perform normalization
        tissue_rgb_normalized = reinhard(crop, 
                                        target_mu=cnorm['mu'], 
                                        target_sigma=cnorm['sigma'],
                                        mask_out = mask_out)
    
        # Clipping values that go out of range [0, 1]
        #tissue_rgb_normalized = np.clip(tissue_rgb_normalized, 0, 1)
        
        # pasting back the normalised image as an array
        wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = tissue_rgb_normalized
        
        # Convert to PIL and save
        img_normed_pil = Image.fromarray((wsi_array).astype('uint8'))
        output_path = os.path.join(output_folder, f"{(filename)}_{normalisation_method}.jpg")
        img_normed_pil.save(output_path)
        
        # Convert the mask to PIL and save
        mask_pil = Image.fromarray((mask_out*255).astype('uint8'))
        output_path_mask = os.path.join(output_folder, "mask_reinhard_wsi.jpg")
        mask_pil.save(output_path_mask)
        
        
    except Exception as e:
        file.write(f"{filename}: {str(e)}\n")
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


