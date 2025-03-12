# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import glob
import math
import os
import json
import numpy as np
import torch


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


def extract_object_embds(tag2class, jina_model):
    class_embs = torch.from_numpy(np.stack([jina_model(text) for text in tag2class.values()]))
    return class_embs

def extract_predicate_class_emb(mask, tag2class, relation_dict, jina_model):
    if type(mask) == dict:
        mask = np.stack([m['segmentation'] for m in mask])
    if type(relation_dict) == list:
        relation_dict_tuple = dict()
        for rel in relation_dict:
            if "affordance" in rel:
                rel['predicates'] = rel['affordance']
            relation_dict_tuple[(rel['s_id'], rel['o_id'])] = rel['predicates']
        relation_dict = relation_dict_tuple
    seg_maps = np.stack([m for i,m in enumerate(mask)])
    with torch.no_grad():
        # contains_comma = [',' in text for text in relation_dict.values()]
        if len(relation_dict) == 0:
            relation_dict = {('0','0'): "none"}
        for k,v in relation_dict.items():
            if type(v) == list:
                relation_dict[k] = ', '.join(v)
        expanded_rels = np.concatenate([np.array(s.split(", ")) for s in relation_dict.values()])
        split_counts = np.array([len(s.split(", ")) for s in relation_dict.values()])
        split_indices = np.repeat(np.arange(len(relation_dict.values())), split_counts)
        
        extended_rels_embs = torch.from_numpy(np.stack([jina_model(text) for text in expanded_rels]).squeeze())
        if extended_rels_embs.dim() == 1:
            extended_rels_embs = extended_rels_embs.unsqueeze(0)
        summed_embs = np.zeros((len(relation_dict.values()), extended_rels_embs.shape[1]))
        np.add.at(summed_embs, split_indices, extended_rels_embs)
        counts = np.bincount(split_indices)
        
        rel_embs = torch.from_numpy(summed_embs / counts[:, None]).to(torch.bfloat16)
        # normalize the embeddings
        rel_embs /= rel_embs.norm(dim=-1, keepdim=True)
       

        # rel_embs = torch.from_numpy(np.stack([jina_model(text) for text in relation_dict.values()]).squeeze())
        # contains_comma = [',' in text for text in relation_dict.values()]
    rel_embds_dict = {k: v for k, v in zip(relation_dict.keys(), rel_embs)}

    obj2sub = {}
    for k,v in relation_dict.items():
        obj2sub.setdefault(k[1], []).append(k[0])

    return rel_embds_dict, seg_maps, obj2sub

def gen_noun_class_img_emb(mask, obj_class_embds, tag2class):
    
    if type(mask) is dict:
        mask = np.stack([m['segmentation'] for m in mask])
    obj_class_img_emb = torch.zeros(*mask[0].shape,512).float()
    for i,tag_id in enumerate(tag2class.keys()):
        try:
            obj_class_img_emb[mask[int(tag_id)-1]] = obj_class_embds[i].detach().cpu().squeeze()
        except:
            # tags sometime are detected incorrectly and not aligned with the mask
            continue
    return obj_class_img_emb

def extract_bert_mask_feature(mask_path, tag2class_path, relation_dict_path, jina_model, img_size=None, regional_pool=True):
    '''Extract per-pixel OpenSeg features.'''
    masks = np.load(mask_path)
    with open(tag2class_path, 'r') as f:
        tag2class = json.load(f)
    with open(relation_dict_path, 'r') as f:
        relation_dict = json.load(f)
        
    relation_dict_tuple = dict()
    for rel in relation_dict:
        if type(rel['s_id']) is list:
            rel['s_id'] = rel['s_id'][0]
        if type(rel['o_id']) is list:
            rel['o_id'] = rel['o_id'][0]
        if 'relationship' in rel:
            rel['predicates'] = rel['relationship']
        if "relationships" in rel:
            rel['predicates'] = rel['relationships']
        relation_dict_tuple[(rel['s_id'], rel['o_id'])] = rel['predicates']
    relation_dict = relation_dict_tuple
    if len(tag2class.keys()) == 0:
        return torch.zeros(1,1,512), torch.zeros(1,512), dict(), masks, torch.zeros(img_size,dtype=torch.uint8), dict()
    obj_class_embds = extract_object_embds(tag2class, jina_model)
    rel_embds_dict, seg_maps, obj2sub = extract_predicate_class_emb(masks, tag2class, relation_dict, jina_model)
    assert (seg_maps == masks).all()
    if img_size is not None:
        masks = torch.nn.functional.interpolate(torch.tensor(masks).unsqueeze(0).float(), size=(img_size), mode='bilinear', align_corners=True).squeeze(0)>0.5
    feat_2d = gen_noun_class_img_emb(masks, obj_class_embds, tag2class)
    
    feat_2d = feat_2d.to(torch.bfloat16)
    segmentation_map = torch.zeros((masks.shape[1], masks.shape[2]), dtype=torch.uint8)
    for i, mask in enumerate(masks):
        segmentation_map[mask] = i+1
    return feat_2d, obj_class_embds, rel_embds_dict, seg_maps, segmentation_map, obj2sub

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

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2]>0 # make sure the depth is in front
            inside_mask = front_mask*inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T


def obtain_intr_extr_matterport(scene):
    '''Obtain the intrinsic and extrinsic parameters of Matterport3D.'''

    img_dir = os.path.join(scene, 'color')
    pose_dir = os.path.join(scene, 'pose')
    intr_dir = os.path.join(scene, 'intrinsic')
    img_names = sorted(glob.glob(img_dir+'/*.jpg'))

    intrinsics = []
    extrinsics = []
    for img_name in img_names:
        name = img_name.split('/')[-1][:-4]

        extrinsics.append(np.loadtxt(os.path.join(pose_dir, name+'.txt')))
        intrinsics.append(np.loadtxt(os.path.join(intr_dir, name+'.txt')))

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    img_names = np.asarray(img_names)

    return img_names, intrinsics, extrinsics

def get_matterport_camera_data(data_path, locs_in, args):
    '''Get all camera view related infomation of Matterport3D.'''

    # find bounding box of the current region
    bbox_l = locs_in.min(axis=0)
    bbox_h = locs_in.max(axis=0)

    building_name = data_path.split('/')[-1].split('_')[0]
    scene_id = data_path.split('/')[-1].split('.')[0]

    scene = os.path.join(args.data_root_2d, building_name)
    img_names, intrinsics, extrinsics = obtain_intr_extr_matterport(scene)

    cam_loc = extrinsics[:, :3, -1]
    ind_in_scene = (cam_loc[:, 0] > bbox_l[0]) & (cam_loc[:, 0] < bbox_h[0]) & \
                    (cam_loc[:, 1] > bbox_l[1]) & (cam_loc[:, 1] < bbox_h[1]) & \
                    (cam_loc[:, 2] > bbox_l[2]) & (cam_loc[:, 2] < bbox_h[2])

    img_names_in = img_names[ind_in_scene]
    intrinsics_in = intrinsics[ind_in_scene]
    extrinsics_in = extrinsics[ind_in_scene]
    num_img = len(img_names_in)

    # some regions have no views inside, we consider it differently for test and train/val
    if args.split == 'test' and num_img == 0:
        print('no views inside {}, take the nearest 100 images to fuse'.format(scene_id))
        #! take the nearest 100 views for feature fusion of regions without inside views
        centroid = (bbox_l+bbox_h)/2
        dist_centroid = np.linalg.norm(cam_loc-centroid, axis=-1)
        ind_in_scene = np.argsort(dist_centroid)[:100]
        img_names_in = img_names[ind_in_scene]
        intrinsics_in = intrinsics[ind_in_scene]
        extrinsics_in = extrinsics[ind_in_scene]
        num_img = 100

    img_names_in = img_names_in.tolist()

    return intrinsics_in, extrinsics_in, img_names_in, scene_id, num_img