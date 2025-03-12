# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import argparse

import numpy as np
from PIL import Image
import base64
from io import BytesIO
from transformers import AutoModel
import json
import os
import glob
import os.path as osp
import pickle
from tqdm import tqdm
from openai import OpenAI

try:
    # semantic sam
    from semantic_sam.BaseModel import BaseModel
    from semantic_sam import build_model
    from semantic_sam.utils.arguments import load_opt_from_config_file
    from datasets.utils.inference_semsam_m2m_auto import inference_semsam_m2m_auto

    semsam_cfg = "models/semantic_sam_only_sa-1b_swinL.yaml"
    semsam_ckpt = "models/swinl_only_sam_many2many.pth"
    opt_semsam = load_opt_from_config_file(semsam_cfg)
    model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()

except ImportError:
    print("Semantic SAM not installed, if you want to use Semantic SAM instead of SAM you need to install it seperately")
# sam
from segment_anything import sam_model_registry
from datasets.utils.inference_sam_m2m_auto import inference_sam_m2m_auto
sam_ckpt = "models/sam_vit_h_4b8939.pth"
model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()


# set seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
# set up cuda for reproducibility
torch.backends.cudnn.deterministic = True

# sam parameters
label_mode = '1'
text_size, hole_scale, island_scale=640,100,100
text, text_part, text_thresh = '','','0.0'
alpha = 0.15
anno_mode = ['Mask', 'Mark']


OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

gpt_client = OpenAI(api_key=OPEN_API_KEY)
gpt_model_name = "gpt-4o"

jina_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).cuda()
jina_encode = lambda x: jina_model.encode(x, task='text-matching', truncate_dim=512)


def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    if "r3scan" in img_path:
        img = img.rotate(-90, expand=True)
    return img

def rotate_mask_outputs(output, masks):
    
    output = np.rot90(output, k=1)
    for i,m in enumerate(masks):
        m['segmentation'] = np.rot90(m['segmentation'], k=1)
        m['bbox'] = [m['bbox'][1], m['bbox'][0], m['bbox'][3], m['bbox'][2]]
    return output, masks

def generate_masks(image, sam_model_name, level=[3]):
    text_size = min(image.size)
    if sam_model_name == 'sam':
        seg_model = model_sam
        output, mask = inference_sam_m2m_auto(seg_model, image, text_size, label_mode, alpha, anno_mode)
    elif sam_model_name == 'semantic-sam':
        seg_model = model_semsam
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output, mask = inference_semsam_m2m_auto(seg_model, image, level, text, text_part, text_thresh, text_size, 
                              hole_scale, island_scale, True, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
    else:
        raise ValueError(f"Model {sam_model_name} not supported")
    return output, mask

def gpt_som_reasoning(som_image, prompt):
    buffered = BytesIO()
    Image.fromarray(som_image).save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = gpt_client.chat.completions.create(
        model=gpt_model_name,
        response_format={ "type": "json_object" },
        temperature=0,
        seed=0,
        messages=[
            {
            "role": "system",
            "content": "You are an articulate assistant designed to output JSON, that describes the objects and their relationships in the image.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",
                        "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ]  
    )
    output_gpt = response.choices[0].message.content
    return output_gpt


def get_objects_dict(output_gpt, masks):
    tag2class = output_gpt['objects']
    updated_masks = []
    for i,m in enumerate(masks):
        if str(i+1) in tag2class.keys():
            updated_masks.append(m)
        else:
            m['segmentation'] = np.zeros_like(m['segmentation'], dtype=bool)
            updated_masks.append(m)
    return tag2class, updated_masks

def get_objects(output_gpt, masks):
    tag2class, updated_masks = get_objects_dict(output_gpt, masks)
    class_embs = torch.from_numpy(np.stack([jina_encode(text) for text in tag2class.values()]))

    return tag2class, class_embs, updated_masks

def get_object_embds(tag2class):
    class_embs = torch.from_numpy(np.stack([jina_encode(text) for text in tag2class.values()]))
    return class_embs


def get_relationships_dict(output_gpt):
    relation_dict = output_gpt['relationships_affordances']
    return relation_dict


def get_noun_class_img_emb(mask, obj_class_embds, tag2class):
    if type(mask) is dict:
        mask = np.stack([m['segmentation'] for m in mask])
    obj_class_img_emb = torch.zeros(*mask[0].shape,512).float()
    for i,tag_id in enumerate(tag2class.keys()):
        obj_class_img_emb[mask[int(tag_id)-1]] = obj_class_embds[i].detach().cpu().squeeze()
    return obj_class_img_emb


def get_predicate_class_emb(mask, tag2class, relation_dict):
    if type(mask) is dict:
        mask = np.stack([m['segmentation'] for m in mask])
    if type(relation_dict) is list:
        relation_dict_tuple = dict()
        for rel in relation_dict:
            relation_dict_tuple[(rel['s_id'], rel['o_id'])] = rel['predicates']
        relation_dict = relation_dict_tuple
    seg_maps = np.stack([m for i,m in enumerate(mask)])
    with torch.no_grad():
        rel_embs = torch.from_numpy(np.stack([jina_encode(text) for text in relation_dict.values()]).squeeze())
    rel_embds_dict = {k: v for k, v in zip(relation_dict.keys(), rel_embs)}

    obj2sub = {}
    for k,v in relation_dict.items():
        obj2sub.setdefault(k[1], []).append(k[0])

    return rel_embds_dict, seg_maps, obj2sub



def generate_gpt_dataset(imgs_dir, output_dir, prompt, sam_model="sam", redo=None, edit=False):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    need_to_redo = []
    for img_path in tqdm(sorted(os.listdir(imgs_dir))):
        if redo and img_path not in redo:
            continue

        img = load_img(os.path.join(imgs_dir, img_path))
        output, masks = generate_masks(img, sam_model)
        try:
            output_gpt = gpt_som_reasoning(output, prompt)
        except Exception:
            print('azure error')
            output_gpt = '{"objects": {}, "relationship": []}'
            
        if 'r3scan' in imgs_dir:
            output, masks = rotate_mask_outputs(output, masks)
        if redo and edit:
            with open(os.path.join(output_dir, f"{img_path}_gpt_output.txt"), 'r') as f:
                output_gpt = f.read()
        with open(os.path.join(output_dir, f"{img_path}_gpt_output.txt"), 'w') as f:
            f.write(output_gpt)
        try:
            structured_gpt_output = json.loads(output_gpt)
            tag2class, masks = get_objects_dict(structured_gpt_output, masks)
            relation_dict = get_relationships_dict(structured_gpt_output)
            
            json.dump(tag2class, open(os.path.join(output_dir, f"{img_path}_tag2class.json"), 'w'))
            json.dump(relation_dict, open(os.path.join(output_dir, f"{img_path}_relation_dict.json"), 'w'))
            np.save(os.path.join(output_dir, f"{img_path}_masks.npy"), np.stack([m['segmentation'] for m in masks]))
        except Exception:
            print(f"Error in {img_path}")
            need_to_redo.append(img_path)
    print(f"Need to redo: {need_to_redo}")
            

    
def encode_gpt_dataset(gpt_output_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    masks_paths = glob.glob(osp.join(gpt_output_dir, "*_masks.npy"))
    masks_paths.sort()
    text_outputs_paths = glob.glob(osp.join(gpt_output_dir, "*_gpt_output.txt"))
    text_outputs_paths.sort()
    tag2classes_paths = glob.glob(osp.join(gpt_output_dir, "*_tag2class.json"))
    tag2classes_paths.sort()
    relation_dicts_paths = glob.glob(osp.join(gpt_output_dir, "*_relation_dict.json"))
    relation_dicts_paths.sort()

    
    for i in tqdm(range(len(masks_paths))):
        with open(text_outputs_paths[i], 'r') as f:
            text_output = f.read()
        with open(tag2classes_paths[i], 'r') as f:
            tag2class = json.load(f)
        with open(relation_dicts_paths[i], 'r') as f:
            relation_dict = json.load(f)
        masks = np.load(masks_paths[i])
        
        obj_class_embds = get_object_embds(tag2class)
        predicate_class_embds, seg_maps, obj2sub = get_predicate_class_emb(masks, tag2class, relation_dict)

        # save embeddings
        torch.save(obj_class_embds, osp.join(output_dir, f"{i}_obj_class_embds.pt"))
        pickle.dump(predicate_class_embds, open(osp.join(output_dir, f"{i}_predicate_class_embds.pkl"), 'wb'))
        np.save(osp.join(output_dir, f"{i}_seg_maps.npy"), seg_maps)
        json.dump(obj2sub, open(osp.join(output_dir, f"{i}_obj2sub.json"), 'w'))

def get_args():
    parser = argparse.ArgumentParser(description="Generate GPT dataset")
    parser.add_argument("--data_dir", type=str, help="path to nerfstudio dataset")
    parser.add_argument("--scene", type=str, default="replica_room0", help="Scene name")
    parser.add_argument("--mode", type=str, default="gpt", help="[gpt, encode]")
    parser.add_argument("--redo", type=str, default=None, help="comma separated list of image names to redo")
    parser.add_argument("--edit", action="store_true", help="Edit the gpt output")
    parser.add_argument("--out_dir", type=str, default="chatgpt")
    parser.add_argument("--rel_types", type=str, default="semantic", help="either [semantic] or [affordance]")
    parser.add_argument("--sam_model", type=str, default="sam", help="either [sam] or [semantic-sam], need to install semantic-sam")
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    data_dir = args.data_dir
    imgs_dir = os.path.join(data_dir, args.scene, "images")
    chatgpt_output_dir = os.path.join(data_dir, args.scene, args.out_dir)
    
    if args.rel_types == "semantic":
        chat_gpt_prompt = """
            1. Object Identification: Identify all objects in the image by their tag. Create a dict that maps tag_id to class_name.

            2. Affordance/Relationship Detection: For every pair of tagged objects that are clearly related, describe the semantic relationships and affordances as a list of dictionaries using the format [s_id: #n1, subject_class: x, o_id: #n2, object_class: y, predicates: [p1, p2, ...]]. For subjects and objects sharing multiple relationships/affordances, concatenate predicates with a comma in the [predicate] field.

            - Avoid generic terms like "next to" for ambiguous relationships. Instead, specify relationships with precise relationships and affordances describing spatial relationships [over/under etc.], comparative relationships [larger/smaller than, similar/same type/color], functional relationships [part of/belonging to, turns on], support relationships [standing on, hanging on, lying on, attached to].
            - Do not use left/right, always use 3D consistant relationships.
            - Always combine a spatial relationship with a semantic, comparative, functional or support relationship using a comma (e.g., [A] [above, lying on] [B]).
            - For symmetrical relationships, include both directions (e.g., [A] [above] [B] and [B] [below] [A]).
            - Even for distant objects highlight if they are [same/similar/same color/same object type]
            Example Output:

            objects = [4: floor, 7: table, 12: chair, ...]

            relationships_affordances = [
                [s_id: 4, subject_class: table, o_id: 7, object_class: floor, predicates: standing on],
                [s_id: 12, subject_class: chair, o_id: 13, object_class: chair, predicates: next to, same as],
                [s_id: 6, subject_class: pillow, o_id: 8, object_class: couch, predicates: belongs to],
                [s_id: 7, subject_class: floor, o_id: 3, object_class: carpet, predicates: under],
                [s_id: 3, subject_class: carpet, o_id: 7, object_class: floor, predicates: above, lying on],
                [s_id: 9, subject_class: table, o_id: 14, object_class: table, predicates: bigger than],
                ...
            ]
        """
    elif args.rel_types == "affordance":
        chat_gpt_prompt = """
            1. Object Identification: Identify all objects in the image by their tag. Create a dict that maps tag_id to class_name.

            2. Inter-object Affordance/Action Detection: For every pair of tagged objects that are clearly have a shared affordance, describe the affordances/actions as a list of dictionaries using the format [s_id: #n1, subject_class: x, o_id: #n2, object_class: y, affordance: [a1, a2, ...]]. For subjects and objects sharing multiple affordances, concatenate affordances with a comma in the [affordance] field.
            - Only state what is observed in the scene, do not invent affordances.
            - For symmetrical affordances, include both directions (e.g., [A] [heats up] [B] and [B] [is being heated up] [A]).
            - Even for distant objects highlight if they have a general affordance like [belongs to] or [can be organized in].
            Example Output:

            objects = [4: lamp, 7: light switch, 12: remote, ...]

            relationships_affordances = [
                [s_id: 7, subject_class: light switch, o_id: 4, object_class: lamp, predicates: turns on],
                [s_id: 12, subject_class: remote, o_id: 13, object_class: TV, predicates: controls],
                [s_id: 6, subject_class: wall socked, o_id: 8, object_class: toaster, predicates: connectes to],
                [s_id: 9, subject_class: shoe, o_id: 14, object_class: shoe rack, predicates: belongs to],
                [s_id: 2, subject_class: stove, o_id: 3, object_class: kettle, predicates: heats up],
                [s_id: 8, subject_class: twol, o_id: 17, object_class: washing machine, predicates: gets cleaned by],
                ...
            ]
        """
    else:
        raise NotImplementedError("not implemented prompting strategy")

    if args.mode == "gpt" or args.mode == "gpt_redo":
        # in case there was a processing mistake and a manual correction
        generate_gpt_dataset(imgs_dir, chatgpt_output_dir, chat_gpt_prompt, sam_model=args.sam_model, redo=args.redo, edit=args.edit)
