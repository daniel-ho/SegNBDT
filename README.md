# High-resolution networks (HRNets) for Semantic Segmentation

## NBDT Notes
- Use PyTorch 1.4 in order to use NBDT code (already included in requirements.txt)

[Correct Pascal Context Links]
https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/scripts/prepare_pcontext.py
https://github.com/zhanghang1989/detail-api

1. Generate saliency maps

<summary>

NBDT
```
for cls in car building vegetation bus sidewalk rider wall bicycle sky traffic_light; do
	python tools/vis_gradcam.py \
			--cfg experiments/cityscapes/vis/vis_seg_hrnet_w18_small_v1_512x1024_tsw10.yaml \
			--vis-mode GradCAMWhole \
			--crop-size 400 \
			--pixel-max-num-random 1 \
			--image-index-range 0 200 1 \
			--nbdt-node-wnids-for ${cls} \
			--crop-for ${cls} \
			--skip-save-npy \
			--target-layers last_layer.3 \
				TEST.MODEL_FILE output/cityscapes/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484_tsw10/best.pth \
				NBDT.USE_NBDT True;
done;
```
baseline
```
for cls in car building vegetation bus sidewalk rider wall bicycle sky traffic_light; do
		python tools/vis_gradcam.py \
				--cfg experiments/cityscapes/vis/vis_seg_hrnet_w18_small_v1_512x1024.yaml \
				--vis-mode OGGradCAM \
				--crop-size 400 \
				--pixel-max-num-random 1 \
				--image-index-range 0 250 1 \
				--crop-for ${cls} \
				--skip-save-npy \
				--target-layers last_layer.3 \
					TEST.MODEL_FILE output/cityscapes/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484_tsw10/best.pth
done;
```

</summary>

2. Generate templates

<summary>

```
nbdt-hierarchy \
		--path graph-induced-hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.json \
		--vis-no-color-leaves \
		--vis-out-fname template \
		--vis-hide f00000030 f00000031 f00000034 \
		--vis-node-conf f00000032 below.dy 300 \
		--vis-node-conf f00000032 below.href '{{ f00000032 }}' \
		--vis-node-conf f00000032 below.label '5. Car? Yes.' \
		--vis-node-conf f00000032 below.sublabel 'Finds headlights, tires' \
		--vis-node-conf f00000033 below.href '{{ f00000033 }}' \
		--vis-node-conf f00000033 below.label '4. Pavement? No.' \
		--vis-node-conf f00000035 below.dy 250 \
		--vis-node-conf f00000035 below.href '{{ f00000035 }}' \
		--vis-node-conf f00000035 below.label '3. Landscape? No.' \
		--vis-node-conf f00000036 below.dy 250 \
		--vis-node-conf f00000036 below.href '{{ f00000036 }}' \
		--vis-node-conf f00000036 below.label '2. Road? No.' \
		--vis-node-conf f00000036 left.href '{{ original }}' \
		--vis-node-conf f00000036 left.label '1. Start here' \
		--vis-node-conf f00000036 left.sublabel 'Goal: Classify center pixel' \
		--vis-zoom 1.75 \
		--vis-color-path-to car \
		--vis-below-dy 375 \
		--vis-scale 0.8 \
		--vis-margin-top -125 \
		--vis-height 500 \
		--vis-width 900
```

```
nbdt-hierarchy \
		--path graph-induced-hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.json \
		--vis-no-color-leaves \
		--vis-out-fname template \
		--vis-hide f00000033 \
		--vis-node-conf f00000034 below.href '{{ f00000034 }}' \
		--vis-node-conf f00000034 below.label '4. Building? Yes.' \
		--vis-node-conf f00000035 below.href '{{ f00000035 }}' \
		--vis-node-conf f00000035 below.label '3. Landscape? Yes.' \
		--vis-node-conf f00000036 below.href '{{ f00000036 }}' \
		--vis-node-conf f00000036 below.label '2. Road? No.' \
		--vis-node-conf f00000036 left.href '{{ original }}' \
		--vis-node-conf f00000036 left.label '1. Start here' \
		--vis-node-conf f00000036 left.sublabel 'Goal: Classify center pixel' \
		--vis-zoom 1.75 \
		--vis-color-path-to building \
		--vis-below-dy 250 \
		--vis-scale 0.8 \
		--vis-margin-top -50 \
		--vis-height 450 \
		--vis-width 800
```

```
nbdt-hierarchy \
		--path graph-induced-hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.json \
		--vis-no-color-leaves \
		--vis-out-fname template \
    --vis-root f00000031 \
		--vis-hide n00002684 f00000028 \
		--vis-node-conf f00000031 below.href '{{ f00000031 }}' \
		--vis-node-conf f00000031 below.label '2. Person or bike? No.' \
		--vis-node-conf f00000031 below.sublabel 'Looks for person, wheel' \
		--vis-node-conf f00000029 below.href '{{ f00000029 }}' \
		--vis-node-conf f00000029 below.label '3. Pole-like? No.' \
		--vis-node-conf n03100490 below.dy 350 \
		--vis-node-conf n03100490 below.href '{{ n03100490 }}' \
		--vis-node-conf n03100490 below.label '4. Truck? No.' \
		--vis-node-conf n04019101 below.href '{{ n04019101 }}' \
		--vis-node-conf n04019101 below.label '5. Bus? Yes.' \
		--vis-node-conf f00000031 left.href '{{ original }}' \
		--vis-node-conf f00000031 left.label '1. Start here' \
		--vis-node-conf f00000031 left.sublabel 'Goal: Classify center pixel' \
		--vis-zoom 1.75 \
		--vis-color-path-to bus \
		--vis-below-dy 250 \
		--vis-scale 0.8 \
		--vis-margin-top -125 \
		--vis-height 500 \
		--vis-width 800
```

```
nbdt-hierarchy \
		--path graph-induced-hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.json \
		--vis-no-color-leaves \
		--vis-out-fname template \
		--vis-hide f00000032 n00001930 f00000034 \
		--vis-node-conf f00000030 below.dy 400 \
		--vis-node-conf f00000030 below.href '{{ f00000030 }}' \
		--vis-node-conf f00000030 below.label '5. Sidewalk? Yes.' \
		--vis-node-conf f00000033 below.dy 350 \
		--vis-node-conf f00000033 below.href '{{ f00000033 }}' \
		--vis-node-conf f00000033 below.label '4. Pavement? Yes.' \
		--vis-node-conf f00000035 below.href '{{ f00000035 }}' \
		--vis-node-conf f00000035 below.label '3. Landscape? No.' \
		--vis-node-conf f00000036 below.href '{{ f00000036 }}' \
		--vis-node-conf f00000036 below.label '2. Road? No.' \
		--vis-node-conf f00000036 left.href '{{ original }}' \
		--vis-node-conf f00000036 left.label '1. Start here' \
		--vis-node-conf f00000036 left.sublabel 'Goal: Classify center pixel' \
		--vis-zoom 1.75 \
		--vis-color-path-to sidewalk \
		--vis-below-dy 250 \
		--vis-scale 0.8 \
		--vis-margin-top -125 \
		--vis-height 500 \
		--vis-width 900
```

```
nbdt-hierarchy \
		--path graph-induced-hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.json \
		--vis-no-color-leaves \
		--vis-out-fname template \
		--vis-hide f00000033 \
		--vis-node-conf f00000034 below.href '{{ f00000034 }}' \
		--vis-node-conf f00000034 below.label '4. Building? No.' \
		--vis-node-conf f00000035 below.href '{{ f00000035 }}' \
		--vis-node-conf f00000035 below.label '3. Landscape? Yes.' \
		--vis-node-conf f00000036 below.href '{{ f00000036 }}' \
		--vis-node-conf f00000036 below.label '2. Road? No.' \
		--vis-node-conf f00000036 left.href '{{ original }}' \
		--vis-node-conf f00000036 left.label '1. Start here' \
		--vis-node-conf f00000036 left.sublabel 'Goal: Classify center pixel' \
		--vis-zoom 1.75 \
		--vis-color-path-to vegetation \
		--vis-below-dy 250 \
		--vis-scale 0.8 \
		--vis-margin-top -100 \
		--vis-height 400 \
		--vis-width 800
```

```
nbdt-hierarchy \
		--path graph-induced-hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.json \
		--vis-no-color-leaves \
		--vis-out-fname template \
		--vis-root f00000031 \
		--vis-hide f00000029 n04576211 \
		--vis-node-conf n00003553 below.dy 250 \
		--vis-node-conf n00003553 below.href '{{ n00003553 }}' \
		--vis-node-conf n00003553 below.label '4. Rider? Yes.' \
		--vis-node-conf n00002684 below.dy 350 \
		--vis-node-conf n00002684 below.href '{{ n00002684 }}' \
		--vis-node-conf n00002684 below.label '3. Cyclist? Yes.' \
		--vis-node-conf f00000031 below.dy 250 \
		--vis-node-conf f00000031 below.href '{{ f00000031 }}' \
		--vis-node-conf f00000031 below.label '2. People? Yes.' \
		--vis-node-conf f00000031 left.href '{{ original }}' \
		--vis-node-conf f00000031 left.label '1. Start here' \
		--vis-node-conf f00000031 left.sublabel 'Goal: Classify center pixel' \
		--vis-zoom 1.75 \
		--vis-color-path-to rider \
		--vis-below-dy 375 \
		--vis-scale 0.8 \
		--vis-margin-top -75 \
		--vis-height 400 \
		--vis-width 900
```

```
nbdt-hierarchy \
		--path graph-induced-hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.json \
		--vis-no-color-leaves \
		--vis-out-fname template \
		--vis-root f00000033 \
		--vis-hide f00000032 f00000034 \
		--vis-node-conf n04341686 below.dy 200 \
		--vis-node-conf n04341686 below.href '{{ n04341686 }}' \
		--vis-node-conf n04341686 below.label '5. Wall? Yes.' \
		--vis-node-conf n00001930 below.dy 250 \
		--vis-node-conf n00001930 below.href '{{ n00001930 }}' \
		--vis-node-conf n00001930 below.label '4. Structure? Yes.' \
		--vis-node-conf f00000030 below.dy 350 \
		--vis-node-conf f00000030 below.href '{{ f00000030 }}' \
		--vis-node-conf f00000030 below.label '3. Verge? No.' \
		--vis-node-conf f00000033 below.href '{{ f00000033 }}' \
		--vis-node-conf f00000033 below.label '2. Pavement? No.' \
		--vis-node-conf f00000033 left.href '{{ original }}' \
		--vis-node-conf f00000033 left.label '1. Start here' \
		--vis-node-conf f00000033 left.sublabel 'Goal: Classify center pixel' \
		--vis-zoom 1.75 \
		--vis-color-path-to wall \
		--vis-below-dy 250 \
		--vis-scale 0.8 \
		--vis-margin-top -75 \
		--vis-height 400 \
		--vis-width 900
```

```
nbdt-hierarchy \
		--path graph-induced-hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.json \
		--vis-no-color-leaves \
		--vis-out-fname template \
		--vis-root f00000031 \
		--vis-hide f00000029 \
		--vis-node-conf n04576211 below.dy 200 \
		--vis-node-conf n04576211 below.href '{{ n04576211 }}' \
		--vis-node-conf n04576211 below.label '5. Bicycle? Yes.' \
		--vis-node-conf n00003553 below.dy 250 \
		--vis-node-conf n00003553 below.href '{{ n00003553 }}' \
		--vis-node-conf n00003553 below.label '4. Rider? No.' \
		--vis-node-conf n00002684 below.dy 350 \
		--vis-node-conf n00002684 below.href '{{ n00002684 }}' \
		--vis-node-conf n00002684 below.label '3. Cyclist? Yes.' \
		--vis-node-conf f00000031 below.dy 250 \
		--vis-node-conf f00000031 below.href '{{ f00000031 }}' \
		--vis-node-conf f00000031 below.label '2. People? Yes.' \
		--vis-node-conf f00000031 left.href '{{ original }}' \
		--vis-node-conf f00000031 left.label '1. Start here' \
		--vis-node-conf f00000031 left.sublabel 'Goal: Classify center pixel' \
		--vis-zoom 1.75 \
		--vis-color-path-to bicycle \
		--vis-below-dy 375 \
		--vis-scale 0.8 \
		--vis-margin-top -75 \
		--vis-height 400 \
		--vis-width 900
```

```
nbdt-hierarchy \
		--path graph-induced-hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.json \
		--vis-no-color-leaves \
		--vis-out-fname template \
    --vis-root f00000031 \
		--vis-hide n03100490 n00002684 n00033020 n00001740 \
		--vis-node-conf f00000031 below.href '{{ f00000031 }}' \
		--vis-node-conf f00000031 below.label '2. Person or bike? No.' \
		--vis-node-conf f00000031 below.sublabel 'Looks for person, wheel' \
		--vis-node-conf f00000031 below.dy 250 \
		--vis-node-conf f00000029 below.href '{{ f00000029 }}' \
		--vis-node-conf f00000029 below.label '3. Pole-like? Yes.' \
		--vis-node-conf f00000029 below.dy 225 \
		--vis-node-conf f00000028 below.href '{{ f00000028 }}' \
		--vis-node-conf f00000028 below.label '4. Sky? Yes.' \
		--vis-node-conf f00000031 left.href '{{ original }}' \
		--vis-node-conf f00000031 left.label '1. Start here' \
		--vis-node-conf f00000031 left.sublabel 'Goal: Classify center pixel' \
		--vis-zoom 1.75 \
		--vis-color-path-to sky \
		--vis-below-dy 200 \
		--vis-scale 0.8 \
		--vis-margin-top -62 \
		--vis-height 325 \
		--vis-width 900
```

```
nbdt-hierarchy \
		--path graph-induced-hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.json \
		--vis-no-color-leaves \
		--vis-out-fname template \
    --vis-root f00000029 \
		--vis-hide n03100490 n00002684 \
		--vis-node-conf f00000029 below.href '{{ f00000029 }}' \
		--vis-node-conf f00000029 below.label '3. Pole-like? Yes.' \
		--vis-node-conf f00000029 below.dy 225 \
		--vis-node-conf f00000028 below.href '{{ f00000028 }}' \
		--vis-node-conf f00000028 below.label '4. Sky? No.' \
		--vis-node-conf n00001740 below.href '{{ n00001740 }}' \
		--vis-node-conf n00001740 below.label '5. Pole? No.' \
		--vis-node-conf n00033020 below.href '{{ n00033020 }}' \
		--vis-node-conf n00033020 below.label '6. Traffic Light? Yes.' \
		--vis-node-conf f00000029 left.href '{{ original }}' \
		--vis-node-conf f00000029 left.label '1. Start here' \
		--vis-node-conf f00000029 left.sublabel 'Goal: Classify center pixel' \
		--vis-zoom 1.75 \
		--vis-color-path-to traffic_light \
		--vis-below-dy 200 \
		--vis-scale 0.8 \
		--vis-margin-top -50 \
		--vis-height 350 \
		--vis-width 900
```

</summary>

3. Generate all figures

```
for cls in car building vegetation bus sidewalk rider wall bicycle sky traffic_light; do python /data/alvinwan/nbdt-segmentation/tools/vis_copy.py template-${cls}.html --dirs-for-cls ${cls} --suffix=-${cls}; done
```

Optionally generate survey

<summary>

```
python /data/alvinwan/nbdt-segmentation/tools/vis_survey.py --baseline `ls oggradcam*crop400/*` --baseline-original `ls oggradcam*original/*` --ours image*.html
```

</summary>

## Branches
- This is the implementation for PyTorch 1.1.
- The HRNet + OCR version ia available [here](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR).
- The PyTorch 0.4.1 version is available [here](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/master).

## News
- HRNet + OCR + SegFix: Rank \#1 (84.5) in [Cityscapes leaderboard](https://www.cityscapes-dataset.com/benchmarks/). OCR: object contextual represenations [pdf](https://arxiv.org/pdf/1909.11065.pdf). ***HRNet + OCR is reproduced [here](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR)***.
- Thanks Google and UIUC researchers. A modified HRNet combined with semantic and instance multi-scale context achieves SOTA panoptic segmentation result on the Mapillary Vista challenge. See [the paper](https://arxiv.org/pdf/1910.04751.pdf).
- Small HRNet models for Cityscapes segmentation. Superior to MobileNetV2Plus ....
- Rank \#1 (83.7) in [Cityscapes leaderboard](https://www.cityscapes-dataset.com/benchmarks/). HRNet combined with an extension of [object context](https://arxiv.org/pdf/1809.00916.pdf)

- Pytorch-v1.1 and the official Sync-BN supported. We have reproduced the cityscapes results on the new codebase. Please check the [pytorch-v1.1 branch](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).

## Introduction
This is the official code of [high-resolution representations for Semantic Segmentation](https://arxiv.org/abs/1904.04514).
We augment the HRNet with a very simple segmentation head shown in the figure below. We aggregate the output representations at four different resolutions, and then use a 1x1 convolutions to fuse these representations. The output representations is fed into the classifier. We evaluate our methods on three datasets, Cityscapes, PASCAL-Context and LIP.

![](figures/seg-hrnet.png)

## Segmentation models
HRNetV2 Segmentation models are now available. All the results are reproduced by using this repo!!!

The models are initialized by the weights pretrained on the ImageNet. You can download the pretrained models from  https://github.com/HRNet/HRNet-Image-Classification.

### Memory usage and time cost
Memory and time cost comparison for semantic segmentation on PyTorch 1.0 in terms of training/inference memory and training/inference time. The numbers for training are obtained on a machine with 4 V100 GPU cards. During training, the input size is 512x1024 and the batch size is 8. The numbers for inference are obtained on a single V100 GPU card. The input size is 1024x2048.

| approach | train mem | train sec./iter |infer. mem | infer sec./image | mIoU |
| :--: | :--: | :--: | :--: | :--: | :--: |
| PSPNet | 14.4G | 0.837| 1.60G | 0.397 | 79.7 |
| DeepLabV3 | 13.3G | 0.850 | 1.15G | 0.411 | 78.5 |
| HRNet-W48 | 13.9G | 0.692 | 1.79G | 0.150 | 81.1 |


### Big models

1. Performance on the Cityscapes dataset. The models are trained and tested with the input size of 512x1024 and 1024x2048 respectively.
If multi-scale testing is used, we adopt scales: 0.5,0.75,1.0,1.25,1.5,1.75.

| model | Train Set | Test Set |#Params | GFLOPs | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W48 | Train | Val | 65.8M | 696.2 | No | No | No | 81.1 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gSlK7Fju_sXCxFUt?e=WZ96Ck)/[BaiduYun(Access Code:t6ri)](https://pan.baidu.com/s/1GXNPm5_DuzVVoKob2pZguA)|

2. Performance on the LIP dataset. The models are trained and tested with the input size of 473x473.

| model |#Params | GFLOPs | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W48 | 65.8M | 74.3 | No | No | Yes | 55.8 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gSjZUHtqfojPfBc6?e=4sE90v)/[BaiduYun(Access Code:sbgy)](https://pan.baidu.com/s/17LAPB-7wsGFPVpHF51tI-w)|

### Small models

The models are initialized by the weights pretrained on the ImageNet. You can download the pretrained models from  https://github.com/HRNet/HRNet-Image-Classification.

Performance on the Cityscapes dataset. The models are trained and tested with the input size of 512x1024 and 1024x2048 respectively. The results of other small models are obtained from Structured Knowledge Distillation for Semantic Segmentation(https://arxiv.org/abs/1903.04197).

| model | Train Set | Test Set |#Params | GFLOPs | OHEM | Multi-scale| Flip | Distillation | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| SQ | Train | Val | - | - | No | No | No | No | 59.8 | |
| CRF-RNN | Train | Val | - | - | No | No | No | No | 62.5 | |
| Dilation10 | Train | Val | 140.8 | - | No | No | No | No | 67.1 | |
| ICNet | Train | Val | - | - | No | No | No | No | 70.6 | |
| ResNet18(1.0) | Train | Val | 15.2 | 477.6 | No | No | No | No | 69.1 | |
| ResNet18(1.0) | Train | Val | 15.2 | 477.6 | No | No | No | Yes | 72.7 | |
| MD(Enhanced) | Train | Val | 14.4 | 240.2 | No | No | No | No | 67.3 | |
| MD(Enhanced) | Train | Val | 14.4 | 240.2 | No | No | No | Yes | 71.9 | |
| MobileNetV2Plus | Train | Val | 8.3 | 320.9 | No | No | No | No | 70.1 | |
| MobileNetV2Plus | Train | Val | 8.3 | 320.9 | No | No | No | Yes | 74.5 | |
| HRNetV2-W18-Small-v1 | Train | Val | 1.5M | 31.1 | No | No | No | No | 70.3 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gSEsg-2sxTmZL2AT?e=AqHbjh)/[BaiduYun(Access Code:63be)](https://pan.baidu.com/s/17pr-he0HEBycHtUdfqWr3g)|
| HRNetV2-W18-Small-v2 | Train | Val | 3.9M | 71.6 | No | No | No | No | 76.2 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gSAL4OurOW0RX4JH?e=ptLwpW)/[BaiduYun(Access Code:k23v)](https://pan.baidu.com/s/155Qxztpc-DU_zmrSOUvS5Q)|

## Quick start
### Install
1. Install PyTorch=1.1.0 following the [official instructions](https://pytorch.org/)
2. git clone https://github.com/HRNet/HRNet-Semantic-Segmentation $SEG_ROOT
3. Install dependencies: pip install -r requirements.txt

If you want to train and evaluate our models on PASCAL-Context, you need to install [details](https://github.com/zhanghang1989/detail-api).
````bash
# PASCAL_CTX=/path/to/PASCAL-Context/
git clone https://github.com/zhanghang1989/detail-api.git $PASCAL_CTX
cd $PASCAL_CTX/PythonAPI
python setup.py install
````

### Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/), [LIP](http://sysu-hcp.net/lip/) and [PASCAL-Context](https://cs.stanford.edu/~roozbeh/pascal-context/) datasets.

Your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│       ├── test
│       ├── train
│       └── val
├── lip
│   ├── TrainVal_images
│   │   ├── train_images
│   │   └── val_images
│   └── TrainVal_parsing_annotations
│       ├── train_segmentations
│       ├── train_segmentations_reversed
│       └── val_segmentations
├── pascal_ctx
│   ├── common
│   ├── PythonAPI
│   ├── res
│   └── VOCdevkit
│       └── VOC2010
├── list
│   ├── cityscapes
│   │   ├── test.lst
│   │   ├── trainval.lst
│   │   └── val.lst
│   ├── lip
│   │   ├── testvalList.txt
│   │   ├── trainList.txt
│   │   └── valList.txt
````

### Train and test
Please specify the configuration file.

For example, train the HRNet-W48 on Cityscapes with a batch size of 12 on 4 GPUs:
````bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````

For example, evaluating our model on the Cityscapes validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating our model on the Cityscapes test set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET list/cityscapes/test.lst \
                     TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating our model on the PASCAL-Context validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/pascal_ctx/seg_hrnet_w48_cls59_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml \
                     TEST.MODEL_FILE hrnet_w48_pascal_context_cls59_480x480.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
                     TEST.FLIP_TEST True
````
Evaluating our model on the LIP validation set with flip testing:
````bash
python tools/test.py --cfg experiments/lip/seg_hrnet_w48_473x473_sgd_lr7e-3_wd5e-4_bs_40_epoch150.yaml \
                     DATASET.TEST_SET list/lip/testvalList.txt \
                     TEST.MODEL_FILE hrnet_w48_lip_cls20_473x473.pth \
                     TEST.FLIP_TEST True \
                     TEST.NUM_SAMPLES 0
````

## Other applications of HRNet
* [Human pose estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
* [Image Classification](https://github.com/HRNet/HRNet-Image-Classification)
* [Object detection](https://github.com/HRNet/HRNet-Object-Detection)
* [Facial landmark detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)

## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng,
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)

## Acknowledgement
We adopt ~~sync-bn implemented by [InplaceABN](https://github.com/mapillary/inplace_abn).~~ the PyTorch official syncbn.

We adopt data precosessing on the PASCAL-Context dataset, implemented by [PASCAL API](https://github.com/zhanghang1989/detail-api).
