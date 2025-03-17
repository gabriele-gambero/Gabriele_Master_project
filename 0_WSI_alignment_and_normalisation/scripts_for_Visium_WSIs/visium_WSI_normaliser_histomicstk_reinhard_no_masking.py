import histomicstk
import os
import argparse
import datetime
import numpy as np
from PIL import Image
from skimage.transform import resize
from histomicstk.preprocessing.color_normalization import reinhard

# Start timing the script execution
starttime = datetime.datetime.now()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Normalize WSI images using Reinhard method without masking')
parser.add_argument('--input_wsi', required=True, help='Path to the input WSI image')
parser.add_argument('--target_image', default=None, help='Path to the target image for normalization, otherwise default values')
args = parser.parse_args()


# setting the working directory
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/0_WSI_alignment_and_normalisation/")
print(os.getcwd())

# ----------------------------------------------------
INPUT_WSI = args.input_wsi
TARGET_IMAGE_PATH = args.target_image
normalisation_method = 'histomicsTK_reinhard_nomasking'

# use_default_target = True   # True if you want to use the default values from function example, False for using the choosen target image 

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
# ----------------------------------------------------

if INPUT_WSI in wsis_dict:
    selected_data = wsis_dict[INPUT_WSI]
    selected_sample = selected_data["SAMPLE"]
    selected_image_version = selected_data["IMAGE_VERSION"]
    selected_cropping_coordinates = selected_data["CROPPING_COORDINATES"]
else:
    print("Error: Image path not found in dataset.")
    exit(1)



if TARGET_IMAGE_PATH:
    # Using a custom target image
    target_temp_path = "target_is_" + TARGET_IMAGE_PATH.split("/")[-1].split(".")[0]
    output_folder = f"./output/{selected_sample}/wsi_{selected_image_version}/{normalisation_method}/{target_temp_path}"
    
    target = Image.open(TARGET_IMAGE_PATH).convert("RGB")
    target_array = np.array(target)
    cnorm = {'mu': target_array.mean(axis=(0, 1)), 'sigma': target_array.std(axis=(0, 1))}
else:
    # Using default target values
    output_folder = f"./output/{selected_sample}/wsi_{selected_image_version}/{normalisation_method}/target_is_default"
    
    cnorm = {'mu': np.array([8.74108109, -0.12440419, 0.0444982]),
             'sigma': np.array([0.6135447, 0.10989545, 0.0286032])}



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
        top_left = selected_cropping_coordinates[0] # (width, height)
        bottom_right = selected_cropping_coordinates[1]

        # Crop the region
        crop = wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        # Perform normalization
        tissue_rgb_normalized = reinhard(crop, 
                                        target_mu=cnorm['mu'], 
                                        target_sigma=cnorm['sigma'])
    
        
        # pasting back the normalised image as an array
        wsi_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = tissue_rgb_normalized
        
        # Convert to PIL and save
        img_normed_pil = Image.fromarray((wsi_array).astype('uint8'))
        output_path = os.path.join(output_folder, f"{(filename)}_{normalisation_method}.jpg")
        img_normed_pil.save(output_path)

        
        
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
