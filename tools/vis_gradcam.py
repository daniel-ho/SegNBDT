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
from utils.gradcam import SegGradCAM, SegNormGrad, GradCAM, SegGradCAMWhole, \
    SegNormGradWhole
from utils.modelsummary import get_model_summary
from utils.utils import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize GradCAM')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--vis-mode', type=str, default='GradCAM',
                        choices=['GradCAM','NormGrad','GradCAMWhole','NormGradWhole'],
                        help='Type of gradient visualization')
    parser.add_argument('--image-index', type=int, default=0,
                        help='Index of input image for GradCAM')
    parser.add_argument('--pixel-i', type=int, default=0, nargs='*',
                        help='i coordinate of pixel from which to compute GradCAM')
    parser.add_argument('--pixel-j', type=int, default=0, nargs='*',
                        help='j coordinate of pixel from which to compute GradCAM')
    parser.add_argument('--pixel-i-range', type=int, default=0, nargs=3,
                        help='Range for pixel i. Expects [start, end) and step.')
    parser.add_argument('--pixel-j-range', type=int, default=0, nargs=3,
                        help='Range for pixel j. Expects [start, end) and step.')
    parser.add_argument('--pixel-cartesian-product', action='store_true',
                        help='Compute cartesian product between all is and js '
                             'for the full list of pixels.')
    parser.add_argument('--suffix', default='',
                        help='Appended to each image filename.')
    parser.add_argument('--target-layers', type=str,
                        help='List of target layers from which to compute GradCAM')
    parser.add_argument('--nbdt-node-wnid', type=str, default='',
                        help='WNID of NBDT node from which to compute output logits')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

class_names = [
    'unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
    'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
    'motorcycle', 'bicycle'
]

def get_pixels(pixel_i, pixel_j, pixel_i_range, pixel_j_range, cartesian_product):
    assert not (pixel_i and pixel_i_range), \
        'Can only specify list of numbers (--pixel-i) OR a range (--pixel-i-range)'
    pixel_is = pixel_i or range(*pixel_i_range)

    assert not (pixel_j and pixel_j_range), \
        'Can only specify list of numbers (--pixel-j) OR a range (--pixel-j-range)'
    pixel_js = pixel_j or range(*pixel_j_range)

    if cartesian_product:
        return sum([ [(i, j) for i in pixel_is] for j in pixel_js ], [])
    return list(zip(pixel_is, pixel_js))

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

def save_gradcam(save_path, gradcam, raw_image, paper_cmap=False,
        minimum=None, maximum=None):
    gradcam = gradcam.cpu().numpy()
    np_save_path = save_path.replace('.jpg', '.npy')
    np.save(np_save_path, gradcam)
    gradcam = GradCAM.normalize_np(gradcam, minimum=minimum, maximum=maximum)[0,0]
    cmap = cm.hot(gradcam)[..., 2::-1] * 255.0
    if paper_cmap:
        alpha = gradcam[..., None]
        gradcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gradcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(save_path, np.uint8(gradcam), [cv2.IMWRITE_JPEG_QUALITY, 50])

def generate_output_dir(output_dir, vis_mode, target_layer, use_nbdt, nbdt_node_wnid):
    vis_mode = vis_mode.lower()
    target_layer = target_layer.replace('model.', '')

    dir = os.path.join(output_dir, f'{vis_mode}_{target_layer}')
    if use_nbdt:
        dir += f'_{nbdt_node_wnid}'
    os.makedirs(dir, exist_ok=True)
    return dir

def generate_save_path(output_dir, gradcam_kwargs, suffix='', ext='jpg'):
    fname = generate_fname(gradcam_kwargs)
    save_path = os.path.join(output_dir, f'{fname}.{ext}')
    return save_path

def generate_fname(kwargs, order=('image', 'pixel_i', 'pixel_j')):
    parts = []
    kwargs = kwargs.copy()
    for key in order:
        if key not in kwargs:
            continue
        parts.append(f'{key}-{kwargs.pop(key)}')
    for key in sorted(kwargs):
        parts.append(f'{key}-{kwargs.pop(key)}')
    return '-'.join(parts)

def compute_overlap(label, gradcam):
    cls_to_mass = {}
    gradcam = GradCAM.normalize(gradcam.data.numpy())[0,0]
    gradcam /= gradcam.sum()
    for cls in map(int, np.unique(label.tolist())):
        cls_to_mass[cls] = gradcam[label == cls].sum()
    return cls_to_mass

def save_overlap(save_path, gradcam, label, k=5):
    overlap = compute_overlap(label, gradcam)
    max_keys = list(sorted(overlap, key=lambda key: overlap[key]))[:k]
    max_labels = [class_names[key] for key in max_keys]
    max_values = [overlap[key] for key in max_keys]
    np.save(save_path, overlap)
    plt.barh(max_labels, max_values)

    save_path = save_path.replace('.npy', '.jpg')
    plt.savefig(save_path)

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
    image, label, _, name = test_dataset[args.image_index]
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

    # Append model. to target layers if using nbdt
    if config.NBDT.USE_NBDT:
        target_layers = ['model.' + layer for layer in target_layers]

    # Run forward pass once, outside of loop
    if config.NBDT.USE_NBDT:
        logger.info("Using logits from node with wnid {}...".format(args.nbdt_node_wnid))
    Saliency = eval('Seg'+args.vis_mode)   # change to dict?
    gradcam = Saliency(model=model, candidate_layers=target_layers,
        use_nbdt=config.NBDT.USE_NBDT, nbdt_node_wnid=args.nbdt_node_wnid)
    pred_probs, pred_labels = gradcam.forward(image)

    maximum, minimum = -1000, 0
    logger.info(f'=> Starting bounds: ({minimum}, {maximum})')

    def generate_and_save_saliency():
        """too lazy to move out to global lol"""
        nonlocal maximum, minimum, label
        # Generate GradCAM + save heatmap
        heatmaps = []
        raw_image = retrieve_raw_image(test_dataset, args.image_index)
        for layer in target_layers:
            gradcam_region = gradcam.generate(target_layer=layer, normalize=False)

            maximum = max(float(gradcam_region.max()), maximum)
            minimum = min(float(gradcam_region.min()), minimum)
            logger.info(f'=> Bounds: ({minimum}, {maximum})')

            gradcam_kwargs['max'] = float('%.3g' % maximum)

            heatmaps.append(gradcam_region)
            output_dir = generate_output_dir(final_output_dir, args.vis_mode, layer, config.NBDT.USE_NBDT, args.nbdt_node_wnid)
            save_path = generate_save_path(output_dir, gradcam_kwargs)
            logger.info('Saving {} heatmap at {}...'.format(args.vis_mode, save_path))
            save_gradcam(save_path, gradcam_region, raw_image, minimum=minimum, maximum=maximum)

            output_dir += '_overlap'
            os.makedirs(output_dir, exist_ok=True)
            save_path = generate_save_path(output_dir, gradcam_kwargs, ext='npy')
            logger.info('Saving {} overlap data at {}...'.format(args.vis_mode, save_path))
            save_path = generate_save_path(output_dir, gradcam_kwargs, ext='jpg')
            logger.info('Saving {} overlap plot at {}...'.format(args.vis_mode, save_path))
            save_overlap(save_path, gradcam_region, label)
        if len(heatmaps) > 1:
            combined = torch.prod(torch.stack(heatmaps, dim=0), dim=0)
            combined /= combined.max()
            save_path = generate_save_path(final_output_dir, args.vis_mode, gradcam_kwargs, 'combined', config.NBDT.USE_NBDT, args.nbdt_node_wnid)
            logger.info('Saving combined {} heatmap at {}...'.format(args.vis_mode, save_path))
            save_gradcam(save_path, combined, raw_image)

    if getattr(Saliency, 'whole_image', False):
        assert not (
                args.pixel_i or args.pixel_j or args.pixel_i_range
                or args.pixel_j_range), \
            'the "Whole" saliency method generates one map for the whole ' \
            'image, not for specific pixels'
        gradcam_kwargs = {'image': args.image_index}
        if args.suffix:
            gradcam_kwargs['suffix'] = args.suffix
        gradcam.backward(pred_labels[:,[0],:,:])

        generate_and_save_saliency()
        return

    pixels = get_pixels(
        args.pixel_i, args.pixel_j, args.pixel_i_range, args.pixel_j_range,
        args.pixel_cartesian_product)
    logger.info(f'Running on {len(pixels)} pixels.')

    for pixel_i, pixel_j in pixels:
        assert pixel_i < test_size[0] and pixel_j < test_size[1], \
            "Pixel ({},{}) is out of bounds for image of size ({},{})".format(
                pixel_i,pixel_j,test_size[0],test_size[1])

        # Run backward pass
        # Note: Computes backprop wrt most likely predicted class rather than gt class
        gradcam_kwargs = {'image': args.image_index, 'pixel_i': pixel_i, 'pixel_j': pixel_j}
        if args.suffix:
            gradcam_kwargs['suffix'] = args.suffix
        logger.info(f'Running {args.vis_mode} on image {args.image_index} at pixel ({pixel_i},{pixel_j}). Using filename suffix: {args.suffix}')
        output_pixel_i, output_pixel_j = compute_output_coord(pixel_i, pixel_j, test_size, pred_probs.shape[2:])
        gradcam.backward(pred_labels[:,[0],:,:], output_pixel_i, output_pixel_j)

        generate_and_save_saliency()

    logger.info(f'=> Final bounds are: ({minimum}, {maximum})')


if __name__ == '__main__':
    main()
