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
    
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/2_image_normalisation/")
print(os.getcwd())

# ----------------------------------------------------
# Set paths
#INPUT_FOLDER = "../1_tiling/output/satac_C1/tiling_output/v3_allspots/tiles_100/"  # Replace with the path to your folder containing images
#INPUT_FOLDER = "../1_tiling/output/visium_2022_FF_WG_10X/tiling_output/img_not_changed_allspots/tiles_100"  # Replace with the path to your folder containing images
INPUT_FOLDER = "../1_tiling/output/visium_FFPE_dcis_idc_10X/tiling_output/img_not_changed_allspots/tiles_100"  # Replace with the path to your folder containing images

tiles_info = INPUT_FOLDER.split('/')

use_default_target = False   # True if you want to use the default values from function example, False for using the choosen target image 
normalisation_method = 'histomicsTK_reinhard_withmasking'
# ----------------------------------------------------


if use_default_target:
    # Setting up the paths for default target
    output_folder = f"./output/{tiles_info[3]}/{tiles_info[5]}/{tiles_info[6]}/{normalisation_method}/target_is_default"
    print(output_folder)
    
    # Using the target values from the function example
    # color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
    cnorm = {'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
             'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),}
    
else:
    # Setting up the paths using a personalised image as target
    TARGET_IMAGE_PATH = "reference_images/reference_full.jpeg"
    target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[1].split(".")[0]
    output_folder = f"./output/{tiles_info[3]}/{tiles_info[5]}/{tiles_info[6]}/{normalisation_method}/{target_temp_path}"
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
    
    for filename in os.listdir(INPUT_FOLDER):
        image_path = os.path.join(INPUT_FOLDER, filename)

        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            tissue_rgb = np.array(img)
            
            # Mask extraction
            mask_out, _ = get_tissue_mask(tissue_rgb, deconvolve_first=True,
                n_thresholding_steps=1, sigma=1.5, min_size=30)
            mask_out = resize(mask_out == 0, output_shape=tissue_rgb.shape[:2],
                    order=0, preserve_range=True) == 1
            
            # Perform normalization
            tissue_rgb_normalized = reinhard(tissue_rgb, 
                                             target_mu=cnorm['mu'], 
                                             target_sigma=cnorm['sigma'], 
                                             mask_out=mask_out)
            
            # Convert to PIL and save
            img_normed_pil = Image.fromarray(tissue_rgb_normalized.astype('uint8')).convert("RGB")
            output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_{normalisation_method}.jpg")
            img_normed_pil.save(output_path)
            
        except Exception as e:
            file.write(f"{filename}: {str(e)}\n")

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

