#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python compute_metrics.py --root_path my_data --eval_model_name my_model --tasks recon gen opt