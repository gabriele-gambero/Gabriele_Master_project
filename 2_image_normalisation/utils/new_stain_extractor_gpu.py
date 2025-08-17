import torch
from torchvahadane.utils import convert_RGB_to_OD, convert_RGB_to_OD_cpu, TissueMaskException
import cv2
import numpy as np
from torchvahadane.dict_learning import dict_learning
use_kornia=True
try:
    from kornia.color import rgb_to_lab 
except ImportError:
    use_kornia = False

class StainExtractorGPU():

    def __init__(self):
        if not use_kornia:
            print('Kornia not installed. Using cv2 fallback')

    def new_rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
        # This is a new version of the Kornia `rgb_to_lab` function. However, I personally couldn't adapt it to my code.
        r"""Convert a RGB image to Lab.

        .. image:: _static/img/rgb_to_lab.png

        The input RGB image is assumed to be in the range of :math:`[0, 1]`. Lab
        color is computed using the D65 illuminant and Observer 2.

        Args:
            image: RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

        Returns:
            Lab version of the image with shape :math:`(*, 3, H, W)`.
            The L channel values are in the range 0..100. a and b are in the range -128..127.

        Example:
            >>> input = torch.rand(2, 3, 4, 5)
            >>> output = rgb_to_lab(input)  # 2x3x4x5

        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

        # Convert from sRGB to Linear RGB
        lin_rgb = rgb_to_linear_rgb(image)

        xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

        # normalize for D65 white point
        xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
        xyz_normalized = torch.div(xyz_im, xyz_ref_white)

        threshold = 0.008856
        power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
        scale = 7.787 * xyz_normalized + 4.0 / 29.0
        xyz_int = torch.where(xyz_normalized > threshold, power, scale)

        x: torch.Tensor = xyz_int[..., 0, :, :]
        y: torch.Tensor = xyz_int[..., 1, :, :]
        z: torch.Tensor = xyz_int[..., 2, :, :]

        L: torch.Tensor = (116.0 * y) - 16.0
        a: torch.Tensor = 500.0 * (x - y)
        _b: torch.Tensor = 200.0 * (y - z)

        out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

        return out
    
    def get_tissue_mask(self, I, luminosity_threshold=0.8):
        """
        Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
        Typically we use to identify tissue in the image and exclude the bright white background.

        uses kornia as optional dependency
        :param I: RGB uint 8 image.
        :param luminosity_threshold: Luminosity threshold.
        :return: Binary mask.
        """
        # ---------------------------- 
        # gabgam has modified the following lines of code. The original ones have been commented.
        #print(f"Shape of input I before processing: {I.shape}")  # Debug input shape

        # Ensure input is in (N, C, H, W) format
        # if I.dim() == 3:  # If batch dimension is missing, add it
        #     I = I.unsqueeze(0)

        if I.shape[0] != 3:  # Ensure 3 channels (RGB)
            raise ValueError(f"Expected 3 channels, got {I.shape[1]}")


        #print(f"Shape of input I after processing: {I.shape}")  # Debug shape after adjustment
        
        if use_kornia:
            #I_LAB = self.new_rgb_to_lab(I[None,:,:,:] / 255.0)
            I_LAB = rgb_to_lab(I[None,:,:,:]/255.0)#.squeeze()
            #print(f"Shape of I_LAB after rgb_to_lab: {I_LAB.shape}")  # Debug shape after adjustment

            #I_LAB = rgb_to_lab(I[None,:,:,:].transpose(1,3)/255)
            L = (I_LAB[:, 0,:,:] /100.0).squeeze() # Convert to range [0,1].
            #print(f"Shape of L after squeezing: {L.shape}")  # Debug shape after adjustment

        # ---------------------------- 
        else:
            I_LAB = torch.from_numpy(cv2.cvtColor(I.cpu().numpy(), cv2.COLOR_RGB2LAB))
            L = (I_LAB[:,:,0]/255.0).squeeze()
        # also check for rgb == 255!
        mask = (L < luminosity_threshold ) & (L > 0)  # fix bug in original stain tools code where black background is not ignored.
        # Check it's not empty
        if mask.sum() == 0:
            raise TissueMaskException("Empty tissue mask computed")
        return mask



    def normalize_matrix_rows(self, A):
        """
        Normalize the rows of an array.
        :param A: An array.
        :return: Array with rows normalized.
        """
        return A / torch.linalg.norm(A, dim=1)[:, None]



    def get_stain_matrix(self, I, luminosity_threshold=0.8, regularizer=0.1, mask=None):
        """
        Stain matrix estimation via method of:
        A. Vahadane et al. 'Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images'

        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param regularizer:
        :return:
        """
        
        # ---------------------------- 
        # gabgam has modified the following lines of code. The original ones have been commented.
        # Ensure input shape is consistent
        # Ensure input shape is consistent
        #print(f"Shape of input I in get_stain_matrix: {I.shape}")
        #print(f"Input shape for get_tissue_mask: {I.shape}")

        # Generate tissue mask
        mask = self.get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,)) if mask is None else  mask.reshape((-1,))
        #.reshape((-1,))
        #print(f"Initial mask shape: {mask.shape}")

        # Convert image to OD space
        OD = convert_RGB_to_OD(I)
        #print(f"OD shape before reshaping: {OD.shape}")

        # Flatten spatial dimensions for both OD and mask
        OD = OD.reshape((-1, 3))
        #mask = mask.reshape(-1)
        #print(f"Mask shape after flattening: {mask.shape}, OD shape after flattening: {OD.shape}")

        # Validate compatibility
        # if mask.shape[0] != OD.shape[0]:
        #     raise ValueError(f"Mask shape {mask.shape} does not match OD shape {OD.shape}")

        # ---------------------------- 
        OD = OD[mask]
        # Change to pylasso dictionary training.
        dictionary, losses = dict_learning(OD, n_components=2, alpha=regularizer, lambd=0.1,
             algorithm='ista', device='cuda', steps=70, constrained=True, progbar=False, persist=True, init='ridge', cut=30, maxiter=50, positive=True)
        # H on first row.
        dictionary = dictionary.T
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]
        return self.normalize_matrix_rows(dictionary)
