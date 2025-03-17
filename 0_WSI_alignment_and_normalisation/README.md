
In this directory, you'll find a description of the WSI alignment and normalisation process.\
The data described here, referred as output of the manual image alignment perfomed on Loupe Browser®, can be found in the `data` folder, in the previous repository of the path to this file. 

# 1. - WSI alignment

***WHY???***

For the WSI alignment, I've used 2 images and generated 3:
1)  the cropped **IHC image**: as **template** for the overlapping
2)  the **full-resolution HE image** for:
      - generating just a full-resolution **fiducial frame**
      - cropping the inside part of the frame containing the tissue obstaining the **cropped tissue**

To perform image processing I've used the **GIMP®** software to help me as the  

The idea is to place the frame on the bottom of all the layers, pasting the IHC black and white image in the center of the frame as template (without touching the fiducial frames) and then try to overlap the real HE-stained slice on top of the previous IHC layer.\
As a final step, the IHC is removed and the final result is a fake HE-stained slice of the original analysed slice that overlaps, as the best compromise, the analysed area referring to common morphological aspects.

# 2. - Manual image alignment on Loupe Browser®

***Describe what is loupe browser and what you did***
\
\
\
\
\
\
\
\
\
\









***Problems***:

Now, they all work with cropping.

**Normalisation with TorchVahadane package**

- `WSI_normaliser_torchvahadane_gpu.py`
Before launching `WSI_normaliser_torch_vahadane_gpu.py` you should also run this script in the terminal:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This allows PyTorch to allocate memory in smaller chunks or "segments," making the memory usage more efficient and reducing fragmentation.
However, it still doesn't work  because the process to try to allocate 25 GB of memory on the GPU, which is 24 GB:

```h

/disk2/work/gabgam/gigi_env/the_project/0_WSI_alignment_and_normalisation

Traceback (most recent call last):

File "/disk2/work/gabgam/gigi_env/the_project/0_WSI_alignment_and_normalisation/WSI_normaliser_torch_vahadane_gpu.py", line 76, in <module>

img = img.permute(2, 0, 1).to(gpu)#.unsqueeze(0)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 25.61 GiB. GPU 0 has a total capacity of 23.64 GiB of which 22.98 GiB is free.

```

- `WSI_normaliser_torchvahadane_cpu.py` once again:
  ```h
  [1]+  Segmentation fault      (core dumped) nohup python3 WSI_normaliser_torch_vahadane_cpu.py > nohup_torchvahadane_cpu.out
  ```

  

**Normalisation with Staintools package**

- `WSI_normaliser_staintools_macenko.py` doesn't work as well because the image is very big and a too big array is created as well.

```h

[1]- Aborted (core dumped) nohup python3 WSI_normaliser_staintools_macenko.py > nohup_staintools_macenko.out

```

New version with cropping: ❗️

  

- `WSI_normaliser_staintools_reinhard.py` It works!!!!!

  

- `WSI_normaliser_staintools_vahadane.py` doesn't work. Problem related to the segmentation process.

```h

[2]+ Segmentation fault (core dumped) nohup python3 WSI_normaliser_staintools_vahadane.py > nohup_staintools_vahadane.out

```

New version with cropping: still doesn't work

**Normalisation with Neural Networks (StainNET)**

- `WSI_normaliser_stainNET.py` has the same problem of the `torchvahadane_gpu` script

- `WSI_normaliser_stainGAN_modelA.py` even if cropped, I got the following error, due to the image size (40000x40000x3) which exceed the 2^32 = 2,147,483,647 maximum number of elements (around 2.1 billion) that can be used as input for the NN.

```
Error processing BCSA4_A2_sATAC_C1_adjacent-Spot000001_v3_newrot_newcrop_realcolors_nofakescaling.jpg: input tensor must fit into 32-bit index math
```

- `WSI_normaliser_stainGAN_modelB.py`: same error as above here.

  

**Normalisation with HistomicsTK package**

  

- `WSI_normaliser_histomicstk_macenko_withmasking.py`: I got the following error, maybe referred to https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian .

```sh

Error processing ../data/spatial_atac/modified_images/BCSA4_A2_sATAC_C1_adjacent-Spot000001_v3_newrot_newcrop_realcolors_nofakescaling: gaussian() got an unexpected keyword argument 'output'

```

  Had to modify the `ski-image.gaussian()`function and replace `output` variable with `out`.

BUT NOW ALL THE HISTOMICSTK ALGORITHMS WORK!!!!!



---
  

Output of `nvidia-smi`, showing that there is enough space for whatever I want to do:
```
Every 2.0s: nvidia-smi                                                                                                                                                                             cbb.medh.ki.se: Mon Jan 20 10:57:32 2025

Mon Jan 20 10:57:32 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Quadro RTX 6000                On  | 00000000:01:00.0 Off |                  Off |
| 34%   31C    P8              13W / 260W |    532MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Quadro RTX 6000                On  | 00000000:81:00.0 Off |                  Off |
| 33%   33C    P8               6W / 260W |      6MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+=======================================================================================+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      6800      G   /usr/lib/xorg/Xorg                            4MiB |
|    0   N/A  N/A   2680690      C   ...forge3/envs/cell2loc_env/bin/python      234MiB |
|    0   N/A  N/A   3331585      C   .../miniforge3/envs/tangram/bin/python      290MiB |
|    1   N/A  N/A      6800      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
```

