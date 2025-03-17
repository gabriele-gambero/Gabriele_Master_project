

nohup python3 WSI_normaliser_staintools_macenko.py > nohup_staintools_macenko.out &
nohup python3 WSI_normaliser_staintools_vahadane.py > nohup_staintools_vahadane.out &
nohup python3 WSI_normaliser_staintools_reinhard.py > nohup_staintools_reinhard.out &

nohup python3 temp_torch_vahadane.py > nohup_temp_torchvahadane_cpu.out&
nohup python3 WSI_normaliser_torch_vahadane_cpu.py > nohup_torchvahadane_cpu.out &
nohup python3 WSI_normaliser_torch_vahadane_gpu.py > nohup_torchvahadane_gpu.out &

nohup python3 WSI_normaliser_stainNET.py > nohup_stainnet.out &
nohup python3 WSI_normaliser_stainGAN_modelA.py > nohup_staingan_A.out &
nohup python3 WSI_normaliser_stainGAN_modelB.py > nohup_staingan_B.out &

nohup python3 WSI_normaliser_histomicstk_macenko_withmasking.py > nohup_histomicstk_macenko_withmasking.out &
nohup python3 WSI_normaliser_histomicstk_macenko_no_masking.py > nohup_histomicstk_macenko_no_masking.out &
nohup python3 WSI_normaliser_histomicstk_reinhard_withmasking.py > nohup_histomicstk_reinhard_withmasking.out &
nohup python3 WSI_normaliser_histomicstk_reinhard_no_masking.py > nohup_histomicstk_reinhard_no_masking.out &





