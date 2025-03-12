# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import gc
import os
import typing


import torch
import numpy as np
from relationfield.data.utils.feature_dataloader import FeatureDataloader
from relationfield.data.utils.siglip_sam_extractor import extract_clip_img_feature
from tqdm import tqdm
import clip
import open_clip
from open_clip import get_tokenizer, create_model_and_transforms
from PIL import Image

sam_version = 'sam'
if sam_version == 'sam2':
    from sam2.build_sam import build_sam2 as build_sam
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
else:
    from segment_anything import SamAutomaticMaskGenerator,sam_model_registry

class ClipSamDataloader(FeatureDataloader):

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
        openseg_cache_path: str = None,
    ):
        assert "image_shape" in cfg
        self.openseg_cache_path = openseg_cache_path
        super().__init__(cfg, device, image_list, cache_path)
    
    def create(self, image_path_list):
        
        pretrained = 'webli'#'laion2b_s32b_b82k'
        model_name = 'ViT-SO400M-14-SigLIP-384'#'ViT-L-14'
        
        clip_model, _, preprocess = create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=self.device)
        sam_model = None
        if sam_version == 'sam2':
            sam_config = build_sam('models/sam2_hiera_l.yaml', 'models/sam2_hiera_large.pt', device=self.device, apply_postprocessing=False)
            sam_model = SAM2AutomaticMaskGenerator(model=sam_config,
                    points_per_side=32,
                    pred_iou_thresh=0.90,
                    stability_score_thresh=0.90,
                    )
        else:
            registry = sam_model_registry['vit_h']
            sam = registry(checkpoint='models/sam_vit_h_4b8939.pth')
            sam = sam.to(device=self.device)
            sam_model = SamAutomaticMaskGenerator(sam,
                    points_per_side=32,
                    pred_iou_thresh=0.90,
                    stability_score_thresh=0.90,
                    )
        

        clip_embeds = []
        for image_id in tqdm(range(len(image_path_list)), desc='clip+sam', total=len(image_path_list), leave=False):
            with torch.no_grad():
                image_path = image_path_list[image_id]
                h = self.cfg['image_shape'][0] // 4
                w = self.cfg['image_shape'][1] // 4
                descriptors = extract_clip_img_feature(image_path, clip_model, preprocess, sam_model, sam_version=sam_version, img_size=[h, w])
            
            clip_embeds.append(descriptors.cpu().detach())

        del clip_model
        del preprocess
        del sam_model
        gc.collect()
        torch.cuda.empty_cache()
        self.data = torch.stack(clip_embeds, dim=0)
        

    def __call__(self, img_points):
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)
