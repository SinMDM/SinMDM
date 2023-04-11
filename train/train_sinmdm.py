# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on a single motion.
"""

import os
import json
import numpy as np
import torch

from data_utils.data_util import load_sin_motion
from utils.fixseed import fixseed
from utils.parser_util import train_sin_args
from utils import dist_util
from train.training_loop import TrainLoop
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation


def main():
    args = train_sin_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("loading motion")
    motion, motion_data = load_sin_motion(args)

    print("creating model and diffusion...")
    args.unconstrained = True
    model, diffusion = create_model_and_diffusion(args, motion_data, motion.shape[0])
    model.to(dist_util.dev())
    diffusion.to(dist_util.dev())

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data=None).run_loop(motion)
    train_platform.close()

if __name__ == "__main__":
    main()
