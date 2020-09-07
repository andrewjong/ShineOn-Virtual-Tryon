import numpy as np
import torch



def segment_cloths_from_image(im, _parse_array):
    """
    from the original image, uses the cloth segmentation map to isolate (crop) the
    cloth-only parts from the image.
    """
    # ISOLATE CLOTH. cloth labels, combines into a 1d binary mask
    from datasets.tryon_dataset import LIP
    _parse_cloth = (
        (_parse_array == LIP.UPPER_CLOTHES).astype(np.float32)
        + (_parse_array == LIP.DRESS).astype(np.float32)
        + (_parse_array == LIP.COAT).astype(np.float32)
    )
    _parse_cloth_mask = torch.from_numpy(_parse_cloth)  # [0,1]
    # upper cloth, segment it from the body
    image_cloth_only = im * _parse_cloth_mask + (1 - _parse_cloth_mask)
    # [-1,1], fill 1 for other parts
    return image_cloth_only

