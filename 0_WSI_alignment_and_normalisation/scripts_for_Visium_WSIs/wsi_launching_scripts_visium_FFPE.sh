
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# for Visium_FFPE_Human_Breast_Cancer_image.tif
# echo "bash /path/to/your_script.sh" | at 00:01 19.02.2025

conda deactivate
conda deactivate
conda activate he_staintools
nohup nice -n 10 python3 visium_WSI_normaliser_staintools_reinhard.py  ../data/visium_ffpe/input_files/Visium_FFPE_Human_Breast_Cancer_image.tif ../2_image_normalisation/reference_images/reference_full.jpeg > nohup_staintools_reinhard.out &
nohup nice -n 10 python3 visium_WSI_normaliser_staintools_reinhard.py  --input_wsi ../data/visium_ffpe/input_files/Visium_FFPE_Human_Breast_Cancer_image.tif --target_image ../2_image_normalisation/reference_images/reference_full.jpeg > nohup_staintools_reinhard.out &


conda deactivate
conda deactivate
conda activate he_histomicstk_39
nohup nice -n 10 python3 visium_WSI_normaliser_histomicstk_macenko_withmasking.py ../data/visium_ffpe/input_files/Visium_FFPE_Human_Breast_Cancer_image.tif ../2_image_normalisation/reference_images/reference_full.jpeg > nohup_histomicstk_macenko_withmasking.out &
nohup nice -n 10 python3 visium_WSI_normaliser_histomicstk_macenko_no_masking.py ../data/visium_ffpe/input_files/Visium_FFPE_Human_Breast_Cancer_image.tif ../2_image_normalisation/reference_images/reference_full.jpeg > nohup_histomicstk_macenko_no_masking.out &
nohup nice -n 10 python3 visium_WSI_normaliser_histomicstk_reinhard_withmasking.py --input_wsi ../data/visium_ffpe/input_files/Visium_FFPE_Human_Breast_Cancer_image.tif --target_image ../2_image_normalisation/reference_images/reference_full.jpeg> nohup_histomicstk_reinhard_withmasking.out &
nohup nice -n 10 python3 visium_WSI_normaliser_histomicstk_reinhard_no_masking.py --input_wsi ../data/visium_ffpe/input_files/Visium_FFPE_Human_Breast_Cancer_image.tif --target_image ../2_image_normalisation/reference_images/reference_full.jpeg> nohup_histomicstk_reinhard_no_masking.out &

