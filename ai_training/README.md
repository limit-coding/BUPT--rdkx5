# AI Training

This folder contains two AI pipelines:

```text
scripts/classification/  Phase 1: CIFAR-100 target-center classification
scripts/yolo11/          Optional YOLO11 detection pipeline
```

## Dataset Source

Competition picture target content comes from CIFAR-100:

```text
https://www.cs.toronto.edu/~kriz/cifar.html
```

The CIFAR data alone is not a complete YOLO detection dataset. It is useful for training the picture-target center classifier. For YOLO detection, capture or synthesize competition-view images and label them.

## Phase 1: Target-Center Classification

This is the first pipeline to run. OpenCV locates/crops the target center, then this model classifies the cropped CIFAR-100 content.

```bash
conda create -n drone-yolo11 python=3.11 -y
conda activate drone-yolo11
pip install -r requirements-yolo11.txt
./scripts/classification/prepare_phase1.sh
./scripts/classification/train_cls.sh
```

Quick smoke-test data generation:

```bash
MAX_TRAIN_PER_CLASS=20 MAX_VAL_PER_CLASS=5 TRAIN_AUGMENTATIONS=1 ./scripts/classification/prepare_phase1.sh
```

## Optional YOLO Detection Classes

```text
0 picture_target
1 special_target
2 ring
3 obstacle
4 landing_h
5 red_light
6 blue_light
```

## Optional YOLO Detection Commands

```bash
conda create -n drone-yolo11 python=3.11 -y
conda activate drone-yolo11
pip install -r requirements-yolo11.txt
python scripts/yolo11/check_dataset.py
./scripts/yolo11/train_det.sh
./scripts/yolo11/val_det.sh
./scripts/yolo11/export_onnx.sh
```

The exported ONNX should then be converted to an RDK X5 BPU `.bin` model.
