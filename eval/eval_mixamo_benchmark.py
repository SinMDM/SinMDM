# This file collect benchmark evaluation abd write it to ClearML
# DO NOT RUN IT STAND-ALONE - Run train/benchmark_training.sh which will run this automatically
import argparse
import time
import os
import numpy as np
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import shutil
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", required=True, type=str, help="")
    parser.add_argument("--num_steps", required=True, type=int, help="")
    parser.add_argument("--ganimator_model", action='store_true', help="")
    parser.add_argument("--benchmark_size", default=10, type=int, help="")
    parser.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                        help="Choose platform to log results. NoPlatform means no logging.")
    args = parser.parse_args()


    # Wait for all evaluation to finish - then read them
    eval_files = [os.path.join(args.benchmark_dir, f'{i:04d}', f'eval_{args.num_steps-1:09d}.npy' if not args.ganimator_model else 'final_eval.npy') for i in range(args.benchmark_size)]  #  FIXME -1
    shutil.copyfile(os.path.join(os.path.dirname(eval_files[0]), 'args.json'), os.path.join(args.benchmark_dir, 'args.json'))
    print(eval_files)
    while not all([os.path.exists(f) for f in eval_files]):
        print('Waiting for all evaluations to finish. DO NOT KILL.')
        print('Missing files: ' + str([f for f in eval_files if not os.path.exists(f)]))
        time.sleep(30)
    eval_dicts = [np.load(e, allow_pickle=True)[None][0] for e in eval_files]

    # Open a clearml task
    with open(os.path.join(args.benchmark_dir, 'args.json'), 'r') as fr:
        orig_args = json.load(fr)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.benchmark_dir + f'-benchmark-{args.num_steps}steps-{args.benchmark_size}samples')
    train_platform.report_args(orig_args, name='Args')

    # Save Results
    for sample_i in range(args.benchmark_size):
        for m, vals in eval_dicts[sample_i].items():
            train_platform.report_scalar(name=f'sample{sample_i:04d}', value=vals['mean'], iteration=args.num_steps,
                                         group_name=m)

    log_file = os.path.join(args.benchmark_dir, 'eval.log')
    metrics_names = list(eval_dicts[0].keys())
    mean_dict = {m: {
        'mean': np.mean([eval_dicts[sample_i][m]['mean'] for sample_i in range(args.benchmark_size)]),
        'std': np.std([eval_dicts[sample_i][m]['mean'] for sample_i in range(args.benchmark_size)]),
    } for m in metrics_names}
    print(mean_dict)
    
    np.save(log_file.replace('.log', '.npy'), mean_dict)
    with open(log_file, 'w') as fw:
        fw.write(str(mean_dict))
    for m, vals in mean_dict.items():
        train_platform.report_scalar(name=f'MEAN', value=vals['mean'], iteration=args.num_steps,
                                     group_name=m)

    train_platform.close()

