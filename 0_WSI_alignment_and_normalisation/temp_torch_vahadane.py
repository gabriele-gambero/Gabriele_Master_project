from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely, or set to a large value like 3_000_000_000
import numpy as np
import os
import torch
import torchvision
from torchvahadane import TorchVahadaneNormalizer
from torchvision.transforms import functional as F

# Set CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Disable PIL image size limit
Image.MAX_IMAGE_PIXELS = None

# setting the working directory
os.chdir("/disk2/user/gabgam/work/gigi_env/the_project/0_WSI_alignment_and_normalisation/")
print(os.getcwd())

# setting a single GPU as the only visible one
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 = first GPU, 1 = second GPU

# ---------------------------------------------------------------------------------
# setting the paths
normalisation_method = 'torch_vahadane_gpu'

# Define paths
INPUT_WSI = "../data/spatial_atac/modified_images/BCSA4_A2_sATAC_C1_adjacent-Spot000001_v3_newrot_newcrop_realcolors_nofakescaling.jpg"
TARGET_IMAGE_PATH = "../2_image_normalisation/reference_images/reference_full.jpeg"


# GPU device
device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

# Load target image
target = Image.open(TARGET_IMAGE_PATH).convert("RGB")
target = torch.Tensor(np.array(target)).permute(2, 0, 1).to(device)

# Initialize TorchVahadaneNormalizer
normalizer = TorchVahadaneNormalizer(device=device, staintools_estimate=False)
normalizer.fit(target)

# Load WSI
wsi = Image.open(INPUT_WSI)
wsi_width, wsi_height = wsi.size

# Define tile size and padding value
tile_size = 1024
fraction = 4
fraction_width = int(round(wsi_width/fraction))
fraction_height = int(round(wsi_height/fraction))

padding_value = 0  # Black padding
# ----------------------------------------------------------
# Iterate over WSI tiles WITH FRACTION
OUTPUT_FOLDER = f"./output/temp_with_torch_vahadane_fraction{fraction}/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for y in range(0, wsi_height, fraction_height):
    for x in range(0, wsi_width, fraction_width):
        # Define the tile region
        tile_box = (x, y, min(x + fraction_width, wsi_width), min(y + fraction_height, wsi_height))
        tile = wsi.crop(tile_box).convert("RGB")

        # Pad the tile if it is not of size 1024x1024
        tile_width, tile_height = tile.size
        if tile_width < fraction_width or tile_height < fraction_height:
            pad_left = 0
            pad_top = 0
            pad_right = fraction_width - tile_width
            pad_bottom = fraction_height - tile_height
            tile = F.pad(torch.Tensor(np.array(tile)).permute(2, 0, 1), (0, pad_right, 0, pad_bottom), fill=padding_value)
        else:
            tile = torch.Tensor(np.array(tile)).permute(2, 0, 1)
# ----------------------------------------------------------
# # Iterate over WSI tiles
# OUTPUT_FOLDER = "./output/temp_with_torch_vahadane/"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# for y in range(0, wsi_height, tile_size):
#     for x in range(0, wsi_width, tile_size):
#         # Define the tile region
#         tile_box = (x, y, min(x + tile_size, wsi_width), min(y + tile_size, wsi_height))
#         tile = wsi.crop(tile_box).convert("RGB")

#         # Pad the tile if it is not of size 1024x1024
#         tile_width, tile_height = tile.size
#         if tile_width < tile_size or tile_height < tile_size:
#             pad_left = 0
#             pad_top = 0
#             pad_right = tile_size - tile_width
#             pad_bottom = tile_size - tile_height
#             tile = F.pad(torch.Tensor(np.array(tile)).permute(2, 0, 1), (0, pad_right, 0, pad_bottom), fill=padding_value)
#         else:
#             tile = torch.Tensor(np.array(tile)).permute(2, 0, 1)
# ----------------------------------------------------------
        # Normalize the tile
        tile = tile.to(device)
        try:
            tile_normalized = normalizer.transform(tile)
            tile_normalized_pil = F.to_pil_image(tile_normalized)

            # Save normalized tile
            output_path = os.path.join(OUTPUT_FOLDER, f"tile_{x}_{y}.jpg")
            tile_normalized_pil.save(output_path)
            print(f"Saved normalized tile: {output_path}")

        except Exception as e:
            print(f"Error processing tile {x}, {y}: {e}")


# Reconstruct WSI from tiles
reconstructed_wsi = Image.new("RGB", (wsi_width, wsi_height))

# Iterate through saved tiles and place them in the right position
for tile_file in os.listdir(OUTPUT_FOLDER):
    if tile_file.endswith(".jpg") and tile_file.startswith("tile_"):
        # Remove the file extension and split the filename
        base_name = os.path.splitext(tile_file)[0]  # Removes ".jpg"
        _, x, y = base_name.split("_")  # Split the base name into parts

        # Convert x and y to integers
        x, y = int(x), int(y)

        # Load the normalized tile
        tile_path = os.path.join(OUTPUT_FOLDER, tile_file)
        tile = Image.open(tile_path)

        # Get the actual size of the tile (in case it was padded)
        tile_width, tile_height = tile.size

        # Paste the tile into the reconstructed WSI
        reconstructed_wsi.paste(tile, (x, y, x + tile_width, y + tile_height))

# Save the reconstructed WSI
reconstructed_output_path = os.path.join(OUTPUT_FOLDER, f"reconstructed_wsi_fraction{fraction}.jpg")
reconstructed_wsi.save(reconstructed_output_path)
print(f"Reconstructed WSI saved to: {reconstructed_output_path}")
