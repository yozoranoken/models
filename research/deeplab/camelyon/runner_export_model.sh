#! /usr/bin/env bash

PYTHON_BIN=$HOME/.anaconda3/envs/camelyon-env/bin/python
PY_SCRIPT=./export_model.py

CHECKPOINT_PATH=$HOME/Projects/model_v2/train/model.ckpt-81174
EXPORT_DIR=$HOME/Projects/model_v2/export

mkdir -p $EXPORT_DIR

EXPORT_PATH=$EXPORT_DIR/frozen_inference_graph.pb


TF_RESEARCH_DIR=$HOME/Projects/tf-models-fork/research
export PYTHONPATH=$PYTHONPATH:$TF_RESEARCH_DIR:$TF_RESEARCH_DIR/slim

$PYTHON_BIN $PY_SCRIPT \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_variant="xception_65" \
    --export_path $EXPORT_PATH \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --crop_size=1280 \
    --crop_size=1280 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4
