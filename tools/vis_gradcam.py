import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval
from utils.gradcam import SegGradCAM, SegNormGrad
from utils.modelsummary import get_model_summary
from utils.utils import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize GradCAM')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--vis-mode', type=str, default='GradCAM', choices=['GradCAM','NormGrad'],
                        help='Type of gradient visualization')
    parser.add_argument('--image-index', type=int, default=0,
                        help='Index of input image for GradCAM')
    parser.add_argument('--pixel-i', type=int, default=0, nargs='*',
                        help='i coordinate of pixel from which to compute GradCAM')
    parser.add_argument('--pixel-j', type=int, default=0, nargs='*',
                        help='j coordinate of pixel from which to compute GradCAM')
    parser.add_argument('--target-layers', type=str,
                        help='List of target layers from which to compute GradCAM')
    parser.add_argument('--nbdt-node', type=str, default='',
                        help='WNID of NBDT node from which to compute output logits')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def compute_output_coord(pixel_i, pixel_j, image_shape, output_shape):
    ratio_i, ratio_j = output_shape[0]/image_shape[0], output_shape[1]/image_shape[1]
    out_pixel_i = int(np.floor(pixel_i * ratio_i))
    out_pixel_j = int(np.floor(pixel_j * ratio_j))
    return out_pixel_i, out_pixel_j

def retrieve_raw_image(dataset, index):
    item = dataset.files[index]
    image = cv2.imread(os.path.join(dataset.root,'cityscapes',item["img"]),
                       cv2.IMREAD_COLOR)
    return image

def save_gradcam(save_path, gradcam, raw_image, paper_cmap=False):
    gradcam = gradcam.cpu().numpy()
    np_save_path = save_path.replace('.png', '.npy')
    np.save(np_save_path, gradcam)
    cmap = cm.jet_r(gradcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gradcam[..., None]
        gradcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gradcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(save_path, np.uint8(gradcam))

def generate_save_path(output_dir, vis_mode, gradcam_args, target_layer, use_nbdt, nbdt_node):
    # TODO: put node in save path
    vis_mode = vis_mode.lower()
    target_layer = target_layer.replace('model.', '')
    save_path_args = gradcam_args + [target_layer]
    if use_nbdt:
        save_path_args += [nbdt_node]
        save_path = os.path.join(output_dir,
            '{}-image-{}-pixel_i-{}-pixel_j-{}-layer-{}-nbdt-{}.png'.format(vis_mode, *save_path_args))
    else:
        save_path = os.path.join(output_dir,
            '{}-image-{}-pixel_i-{}-pixel_j-{}-layer-{}.png'.format(vis_mode, *save_path_args))
    return save_path

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'vis_gradcam')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'best.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Wrap original model with NBDT
    if config.NBDT.USE_NBDT:
        from nbdt.model import SoftSegNBDT
        model = SoftSegNBDT(config.NBDT.DATASET, model, hierarchy=config.NBDT.HIERARCHY)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    # Retrieve input image corresponding to args.image_index
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)
    image,_,_,name = test_dataset[args.image_index]
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    logger.info("Using image {}...".format(name))

    # Define target layer as final convolution layer if not specified
    if args.target_layers:
        target_layers = args.target_layers.split(',')
    else:
        for name, module in list(model.named_modules())[::-1]:
            if isinstance(module, nn.Conv2d):
                target_layers = [name]
            break
    logger.info('Target layers set to {}'.format(str(target_layers)))

    for pixel_i, pixel_j in zip(args.pixel_i, args.pixel_j):
        assert pixel_i < test_size[0] and pixel_j < test_size[1], \
            "Pixel ({},{}) is out of bounds for image of size ({},{})".format(
                pixel_i,pixel_j,test_size[0],test_size[1])

        # Run forward + backward passes
        # Note: Computes backprop wrt most likely predicted class rather than gt class
        gradcam_args = [args.image_index, pixel_i, pixel_j]
        logger.info('Running {} on image {} at pixel ({},{})...'.format(args.vis_mode, *gradcam_args))
        if config.NBDT.USE_NBDT:
            logger.info("Using logits from node with wnid {}...".format(args.nbdt_node))
        gradcam = eval('Seg'+args.vis_mode)(model=model, candidate_layers=target_layers, 
            use_nbdt=config.NBDT.USE_NBDT, nbdt_node=args.nbdt_node)
        pred_probs, pred_labels = gradcam.forward(image)
        pixel_i, pixel_j = compute_output_coord(pixel_i, pixel_j, test_size, pred_probs.shape[2:])
        gradcam.backward(pred_labels[:,[0],:,:], pixel_i, pixel_j)

        # Generate GradCAM + save heatmap
        heatmaps = []
        raw_image = retrieve_raw_image(test_dataset, args.image_index)
        for layer in target_layers:
            gradcam_region = gradcam.generate(target_layer=layer)[0,0]
            heatmaps.append(gradcam_region)
            save_path = generate_save_path(final_output_dir, args.vis_mode, gradcam_args, layer, config.NBDT.USE_NBDT, args.nbdt_node)
            logger.info('Saving {} heatmap at {}...'.format(args.vis_mode, save_path))
            save_gradcam(save_path, gradcam_region, raw_image)
        if len(heatmaps) > 1:
            combined = torch.prod(torch.stack(heatmaps, dim=0), dim=0)
            combined /= combined.max()
            save_path = generate_save_path(final_output_dir, args.vis_mode, gradcam_args, 'combined', config.NBDT.USE_NBDT, args.nbdt_node)
            logger.info('Saving combined {} heatmap at {}...'.format(args.vis_mode, save_path))
            save_gradcam(save_path, combined, raw_image)


if __name__ == '__main__':
    main()
