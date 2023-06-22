# This file collect benchmark evaluation abd write it to ClearML
# DO NOT RUN IT STAND-ALONE - Run train/benchmark_training.sh which will run this automatically
import argparse
import time
import os
import numpy as np
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import shutil
import json
import glob
import re


def fix_intra(args, special_suffix):
    assert args.fix_intra
    all_eval_files = glob.glob(os.path.join(args.benchmark_dir, '*', f'eval_*{special_suffix}.npy'))
    # all_eval_files = glob.glob(os.path.join(args.benchmark_dir, '*', 'final_eval.npy'))  # ganimator only
    for eval_file in all_eval_files:
        vals = np.load(eval_file, allow_pickle=True)[None][0]
        vals['intra_diversity_gt_diff'] = {}
        vals['intra_diversity_gt_diff']['mean'] = \
            abs(vals['intra_diversity_dist']['mean'] - vals['gt_intra_diversity_dist']['mean'])
        vals['intra_diversity_gt_diff']['std'] = 0
        name_fix = eval_file.replace('.npy', '_fix_intra.npy')
        assert not os.path.exists(name_fix)
        np.save(name_fix, vals)
        with open(name_fix.replace('.npy', '.log'), 'w') as fw:
            fw.write(str(vals))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", required=True, type=str, help="")
    parser.add_argument("--num_steps", required=True, type=int, help="")
    parser.add_argument("--benchmark_size", default=10, type=int, help="")
    parser.add_argument("--train_platform_type",
                        default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'],
                        type=str,
                        help="Choose platform to log results. NoPlatform means no logging.")
    parser.add_argument("--fix_intra", action='store_true',
                        help="fix the intra diversity according to the eval_special flag")
    args = parser.parse_args()

    last_reported_iter = args.num_steps - 1
    # last_reported_iter = 15000  # ganimator only

    # Wait for all evaluation to finish - then read them
    last_eval_files = [os.path.join(args.benchmark_dir, f'{i:04d}', f'eval_{last_reported_iter:09d}.npy') for i in
                       range(args.benchmark_size)]  # FIXME -1
    shutil.copyfile(os.path.join(os.path.dirname(last_eval_files[0]), 'args.json'),
                    os.path.join(args.benchmark_dir, 'args.json'))
    print(last_eval_files)

    if not args.fix_intra:
        while not all([os.path.exists(f) for f in last_eval_files]):
            print('Waiting for all evaluations to finish. DO NOT KILL.')
            print('Missing files: ' + str([f for f in last_eval_files if not os.path.exists(f)]))
            time.sleep(30)
    else:
        special_suffix = ''
        fix_intra(args, special_suffix)
        special_suffix += '_fix_intra'

    # Open a clearml task
    with open(os.path.join(args.benchmark_dir, 'args.json'), 'r') as fr:
        orig_args = json.load(fr)
    train_platform_type = eval(args.train_platform_type)
    task_name = args.benchmark_dir + f'-benchmark-{args.num_steps}steps-{args.benchmark_size}samples{special_suffix}'
    train_platform = train_platform_type(task_name)
    train_platform.report_args(orig_args, name='Args')

    # add reports from previous iterations, if exist
    all_eval_files = glob.glob(os.path.join(args.benchmark_dir, '*', f'eval_*{special_suffix}.npy'))
    # all_eval_files = glob.glob(os.path.join(args.benchmark_dir, '*', f'final_eval{special_suffix}.npy'))  # ganimator only
    eval_dicts = {}
    for eval_file in all_eval_files:
        iter = re.findall('.*eval_(\d+).*.npy', eval_file)
        # iter = [last_reported_iter] # ganimator only
        assert len(iter) == 1
        iter = int(iter[0])
        sample_i = re.findall('.*\/(\d\d\d\d)\/.*', eval_file)[0]
        vals = np.load(eval_file, allow_pickle=True)[None][0]
        if sample_i not in eval_dicts.keys():
            eval_dicts[sample_i] = dict()
        eval_dicts[sample_i][iter] = vals

    # Report Results
    for sample_i in range(args.benchmark_size):
        for iter in eval_dicts[f'{sample_i:04d}']:
            for m, vals in eval_dicts[f'{sample_i:04d}'][iter].items():
                train_platform.report_scalar(name=f'sample{sample_i:04d}', value=vals['mean'], iteration=iter,
                                             group_name=m)

    log_file = os.path.join(args.benchmark_dir, f'eval{special_suffix}.log')
    metric_names = eval_dicts['0000'][last_reported_iter].keys()
    mean_dict = dict()

    for iter in eval_dicts['0000'].keys():
        mean_dict[iter] = {m: {'mean': np.mean(
            [eval_dicts[f'{sample_i:04d}'][iter][m]['mean'] for sample_i in range(args.benchmark_size)]),
                               'std': np.std([eval_dicts[f'{sample_i:04d}'][iter][m]['mean'] for sample_i in
                                              range(args.benchmark_size)]),
                               } for m in metric_names}
    print(mean_dict[last_reported_iter])

    np.save(log_file.replace('.log', '.npy'), mean_dict)
    with open(log_file, 'w') as fw:
        fw.write(str(mean_dict))
    for iter in eval_dicts['0000'].keys():
        for m, vals in mean_dict[iter].items():
            train_platform.report_scalar(name=f'MEAN', value=vals['mean'], iteration=iter, group_name=m)

    train_platform.close()
