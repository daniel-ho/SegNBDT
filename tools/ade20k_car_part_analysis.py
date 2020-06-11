import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from nbdt.model import SoftSegNBDT, HardSegNBDT
import _init_paths
import models
import datasets
from config import config
from config import update_config
from torch.nn import functional as F
from nbdt.utils import (coerce_tensor, uncoerce_tensor)
from ade20k_analysis_helper import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description='ADE20K Node Hypotheses')
    parser.add_argument('--cfg',
                        help='ADE20K experiment configuration file',
                        required=True,
                        type=str)
    parser.add_argument('--index',
                        help='index of ADE20K image to run annalysis on',
                        required=True,
                        type=int)
    parser.add_argument('--wnid',
                        help='wnid of ADE20K node to run analysis on',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def nbdt_inference(images, model1, model2, node):
    images = images.float().unsqueeze(0).permute(0, 3, 1, 2)
    size = images.size()
    outputs = model1(images)
    
    if outputs.size()[-2] != size[-2] or outputs.size()[-1] != size[-1]:
        outputs = F.upsample(outputs, (size[-2], size[-1]), mode='bilinear')

    n,c,h,w = outputs.shape
    coerced_outputs = coerce_tensor(outputs)
    node_logits = model2.rules.get_node_logits(coerced_outputs, node)
    logits = node_logits.reshape(n,h,w,node_logits.shape[-1]).permute(0,3,1,2)
    _,labels =  logits.max(1)
    return labels.numpy()

def main():
    args = parse_args()
    
    # build model
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)
    model_state_file = config.TEST.MODEL_FILE
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model1 = model.eval() 

    model2 = HardSegNBDT(dataset=config.NBDT.DATASET,
                         model=model1,
                         hierarchy=config.NBDT.HIERARCHY)
    
    model2 = model2.eval()
    node = model2.rules.wnid_to_node[args.wnid]

    ade = Dataset()

    index = args.index
    print(ade.filename(index)) 
    image = ade.image(index)
    labels = nbdt_inference(torch.from_numpy(image), model1, model2, node)
    obj_masks, masks = ade.get_masks(index)
    
    car_mask = obj_masks[401]
    print("Car Proportion: " + str(np.mean(1-car_mask)))

    for obj in masks:
        print(ade.object_name(obj))
        mask = masks[obj]

        print("Proportion: " + str(np.mean(1-mask)))

        combined_mask = np.uint8(np.equal(car_mask, mask))    
        masked_img = ade.mask_image(image, mask)
        pred = nbdt_inference(torch.from_numpy(masked_img), model1, model2, node)

        masked_preds = ma.masked_array(np.array(pred), combined_mask)
        masked_labels = ma.masked_array(labels, combined_mask)
        acc = np.equal(masked_preds, masked_labels).mean()
        print(acc)

if __name__ == '__main__':
    main()
