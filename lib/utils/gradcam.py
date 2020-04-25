#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class _BaseWrapper(object):

    def __init__(self, model, use_nbdt=False):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.use_nbdt = use_nbdt

    def _encode_one_hot(self, labels):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, labels, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        if self.use_nbdt:
            outputs = self.model.model(image)
            # TODO: just for testing
            node = np.random.choice(self.model.rules.nodes)
            node_logits = self.model.rules.get_node_logits(outputs,node)
            self.logits = node_logits
        else:
            self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, labels):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(labels)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None, use_nbdt=False):
        super(GradCAM, self).__init__(model, use_nbdt)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


class _SegBaseWrapper(_BaseWrapper):

    def _encode_one_hot(self, labels, pixel_i, pixel_j):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        scattered = one_hot[:,:,pixel_i,pixel_j].scatter_(1, labels[:,:,pixel_i,pixel_j], 1.0)
        one_hot[:,:,pixel_i,pixel_j] = scattered
        return one_hot

    def backward(self, labels, pixel_i, pixel_j):
        """
        Class and pixel specific backpropagation
        """
        one_hot = self._encode_one_hot(labels, pixel_i, pixel_j)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)


class SegGradCAM(_SegBaseWrapper, GradCAM):

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = grads

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam
