#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-runs/micro_drone/yolo11n_cifar100_cls/weights/best.pt}"
SOURCE="${SOURCE:-datasets/cifar100_target_cls/val}"
IMGSZ="${IMGSZ:-224}"

yolo classify predict \
  model="$MODEL" \
  source="$SOURCE" \
  imgsz="$IMGSZ" \
  save=True
