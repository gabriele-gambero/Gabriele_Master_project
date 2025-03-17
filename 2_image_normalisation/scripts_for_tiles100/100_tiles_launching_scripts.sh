
conda deactivate
conda deactivate
conda activate he_histomicstk_39
nohup nice -n 10 python3 100_normaliser_histomicstk_macenko_withmasking.py > nohup_htk_macenko_withmasking.out &
nohup nice -n 10 python3 100_normaliser_histomicstk_macenko_nomasking.py > nohup_htk_macenko_nomasking.out
nohup nice -n 10 python3 100_normaliser_histomicstk_reinhard_no_masking.py > nohup_htk_reinhard_nomasking.out &
nohup nice -n 10 python3 100_normaliser_histomicstk_reinhard_withmasking.py > nohup_htk_reinhard_withmasking.out &

conda deactivate
conda deactivate
conda activate he_staintools
nohup nice -n 10 python3 100_normaliser_staintools_macenko.py > nohup_staintools_macenko.out &
nohup nice -n 10 python3 100_normaliser_staintools_vahadane.py > nohup_staintools_vahadane.out &
nohup nice -n 10 python3 100_normaliser_staintools_reinhard.py > nohup_staintools_reinhard.out &


conda deactivate
conda deactivate
conda activate he_torchvahadane
nohup nice -n 10 python3 100_normaliser_torch_vahadane_cpu.py > nohup_torchvahadane_cpu.out &
nohup nice -n 10 python3 100_normaliser_torch_vahadane_gpu.py > nohup_torchvahadane_gpu.out &

# no StainNET or StainGAN here because on the notebook it's fast enough 