# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import torch
import numpy as np
from torchvision import transforms
from datasets.utils.visualizer import Visualizer
from typing import Tuple
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import io
from segment_anything import SamAutomaticMaskGenerator



def inference_sam_m2m_auto(model, image, text_size, label_mode='1', alpha=0.1, anno_mode=['Mask']):
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)
    image_ori = np.asarray(image_ori)

    mask_generator = SamAutomaticMaskGenerator(model)
    outputs = mask_generator.generate(image_ori)


    visual = Visualizer(image_ori)
    sorted_anns = sorted(outputs, key=(lambda x: x['area']), reverse=True)
    label = 1
    # for ann in sorted_anns:
    #     mask = ann['segmentation']
    #     color_mask = np.random.random((1, 3)).tolist()[0]
    #     # color_mask = [int(c*255) for c in color_mask]
    #     demo = visual.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
    #     label += 1
    # im = demo.get_image()

    mask_map = np.zeros(image_ori.shape, dtype=np.uint8)    
    sorted_anns = sorted_anns[:40]
    for i, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]
        # color_mask = [int(c*255) for c in color_mask]
        demo = visual.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
        # assign the mask to the mask_map
        mask_map[mask == 1] = label
        label += 1
        
    im = demo.get_image()    
    # fig=plt.figure(figsize=(10, 10))
    # plt.imshow(image_ori)
    # show_anns(outputs)
    # fig.canvas.draw()
    # im=Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    return im, sorted_anns


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))