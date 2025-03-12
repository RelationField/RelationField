# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import math
import os
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision

process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((384, 384)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
                
            ]
        )

def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
                    intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def mask2segmap(masks, image, preprocessing = None):
    seg_img_list = []
    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    for i in range(len(masks)):
        mask = masks[i]
        seg_img = get_seg_img(mask, image)
        pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))#(384,384))
        if preprocessing is not None:
            pad_seg_img = preprocessing(Image.fromarray(pad_seg_img)).permute(1,2,0).numpy()
        seg_img_list.append(pad_seg_img)

        seg_map[masks[i]['segmentation']] = i
    seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
    seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

    return seg_imgs, seg_map

def extract_clip_img_feature(image_path, clip_model, preprocessing, sam_model, sam_version='sam2', img_size=None, regional_pool=True, device='cuda'):
    '''Extract per-pixel CLIP features.'''
    
    image = Image.open(image_path)
    with torch.no_grad():
        im = np.array(image.convert("RGB"))
        if sam_version == 'sam2':
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks = sam_model.generate(im)
        else:
            masks = sam_model.generate(im)
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    # # filter masks with area less than 20
    # sorted_masks = [mask for mask in sorted_masks if mask['area'] > 20]
    bboxes = [mask['bbox'] for mask in sorted_masks]
    bboxes = [[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] for bbox in bboxes]
    
    seg_imgs, segmaps = mask2segmap(sorted_masks, np.array(image))
    # seg_imgs = [preprocessing(im) for im in seg_imgs]
    with torch.no_grad():
        seg_imgs = process(seg_imgs)
        clip_embed = clip_model.encode_image(seg_imgs)
        # clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds = clip_embed.detach().cpu()
    
    
    clip_masks = torch.zeros((*image.size[::-1],1152), dtype=torch.float32)
    for i in range(len(masks)):
        seg = sorted_masks[i]['segmentation']
        clip_masks[seg.astype(bool)] = clip_embeds[i].detach().cpu().float()
        
    del clip_embeds
    del clip_embed
    del sorted_masks
    del masks

    with torch.no_grad():
        assert torch.isnan(clip_masks).sum() == 0, 'nan in clip_masks'
    if img_size is not None:
        clip_masks = clip_masks.permute(2, 0, 1).unsqueeze(0)
        clip_masks = torch.nn.functional.interpolate(clip_masks, size=img_size, mode='nearest').to(dtype=torch.float16) #, align_corners=True
        clip_masks = clip_masks.squeeze(0).permute(1, 2, 0)
    else:
        clip_masks = clip_masks.to(dtype=torch.float16)
    
    return clip_masks

def save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id, args):
    '''Save features.'''
    
    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points

        rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

        mask_entire = torch.zeros(n_points, dtype=torch.bool)
        mask_entire[rand_ind] = True
        mask = torch.zeros(n_points, dtype=torch.bool)
        mask[point_ids] = True
        mask_entire = mask_entire & mask

        torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                    "mask_full": mask_entire
        },  os.path.join(out_dir, scene_id +'_%d.pt'%(n)))
        print(os.path.join(out_dir, scene_id +'_%d.pt'%(n)) + ' is saved!')


class PointCloudToImageMapper(object):
    def __init__(self, image_dim,
            visibility_threshold=0.25, cut_bound=0, intrinsics=None):
        
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None: # global intrinsics
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int) # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < self.image_dim[0]-self.cut_bound) \
                    * (pi[1] < self.image_dim[1]-self.cut_bound)
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                    - p[2][inside_mask]) <= \
                                    self.vis_thres * depth_cur

            inside_mask[inside_mask] = occlusion_mask
        else:
            front_mask = p[2]>0 # make sure the depth is in front
            inside_mask = front_mask*inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T
