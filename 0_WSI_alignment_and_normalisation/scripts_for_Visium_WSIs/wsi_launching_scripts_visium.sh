conda deactivate


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# nohup python3 visium_WSI_normaliser_staintools_macenko.py ../data/visium_2022_FF_WG/input_files/Visium_Human_Breast_Cancer_image.tif ../2_image_normalisation/reference_images/reference_full.jpeg > nohup_staintools_macenko.out &
nohup python3 visium_WSI_normaliser_staintools_vahadane.py > nohup_staintools_vahadane.out &
nohup python3 visium_WSI_normaliser_staintools_reinhard.py ../data/visium_2022_FF_WG/input_files/Visium_Human_Breast_Cancer_image.tif ../2_image_normalisation/reference_images/reference_full.jpeg > nohup_staintools_reinhard.out &

nohup python3 visium_WSI_normaliser_torch_vahadane_cpu.py > nohup_torchvahadane_cpu.out &
nohup python3 visium_WSI_normaliser_torch_vahadane_gpu.py > nohup_torchvahadane_gpu.out &

nohup python3 visium_WSI_normaliser_stainNET.py > nohup_stainnet.out &
nohup python3 visium_WSI_normaliser_stainGAN_modelA.py > nohup_staingan_A.out &
nohup python3 visium_WSI_normaliser_stainGAN_modelB.py > nohup_staingan_B.out &

conda activate he_histomicstk_39
nohup python3 visium_WSI_normaliser_histomicstk_macenko_withmasking.py ../data/visium_2022_FF_WG/input_files/Visium_Human_Breast_Cancer_image.tif ../2_image_normalisation/reference_images/reference_full.jpeg > nohup_histomicstk_macenko_withmasking.out &
nohup python3 visium_WSI_normaliser_histomicstk_macenko_no_masking.py ../data/visium_2022_FF_WG/input_files/Visium_Human_Breast_Cancer_image.tif ../2_image_normalisation/reference_images/reference_full.jpeg > nohup_histomicstk_macenko_no_masking.out &
nohup python3 visium_WSI_normaliser_histomicstk_reinhard_withmasking.py --input_wsi ../data/visium_2022_FF_WG/input_files/Visium_Human_Breast_Cancer_image.tif --target_image ../2_image_normalisation/reference_images/reference_full.jpeg> nohup_histomicstk_reinhard_withmasking.out &
nohup python3 visium_WSI_normaliser_histomicstk_reinhard_no_masking.py --input_wsi ../data/visium_2022_FF_WG/input_files/Visium_Human_Breast_Cancer_image.tif --target_image ../2_image_normalisation/reference_images/reference_full.jpeg> nohup_histomicstk_reinhard_no_masking.out &