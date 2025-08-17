IMPORTANT: this it not the real first step of the analysis, but conceptually, it should come before step 1, and that's why it's called step 0. So, before trying to use this code, I advice to refer to step 1.

---

In this directory, you'll find a description of the **Spatial ATAC WSI alignment** and general WSI normalisation process.\
The data described here, referred as output of the manual image alignment perfomed on Loupe Browser®, can be found in the `data` folder, in the previous repository of the path to this file. 

The WSI alignment was needed due to incompatibility of HE staining and the Spatial ATAC protocol on the same sample section. For this reason, I0ve manually overlapped the HE adjacent section with the IHC analysed one.

### 1. - Spatial ATAC WSI alignment

For the WSI alignment, I've used 2 images and generated 3:
1)  the cropped **IHC image**: as **template** for the overlapping
2)  the **full-resolution HE image** for:
      - generating just a full-resolution **fiducial frame**
      - cropping the inside part of the frame containing the tissue obstaining the **cropped tissue**

To perform image processing I've used the **GIMP®** software.

No warping or reduction/enlargment of the image has been performed as we think that it would modify the real architecture of the tissue; only moving and rotation.

The idea is to place the frame on the bottom of all the layers, pasting the IHC black and white image in the center of the frame as template (without touching the fiducial frames) and then try to overlap the real HE-stained slice on top of the previous IHC layer.\
As a final step, the IHC is removed and the final result is a fake HE-stained slice of the original analysed slice that overlaps, as the best compromise, the analysed area referring to common morphological aspects.

### 2. - Manual image alignment on Loupe Browser®

In Loupe Browser®, thanks to the added fiducial frame, I've selected the spots that truly overlap the tissue and saved all the useful output.

The analysis of the Spatial ATAC sample leveraged the resulting files.


----


***Problems and limitations in WSI normalisation***:

WSI normalisation is not a normal practise due to their massive size (GBs).
For this reason, a lot of errors related to resource limitations happened during the process.

In the end, only 3 methods succeded for both the samples: HistomicsTK Macenko's (with and without masking) and Staintools Reinhard's implementation.

