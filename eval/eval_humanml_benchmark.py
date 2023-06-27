# This file collect benchmark evaluation and write it to ClearML
# DO NOT RUN IT STAND-ALONE - Run eval/eval_only_humanml_benchmark.sh which will run this automatically
import argparse
import time
import os
import numpy as np
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import shutil
import json
import glob
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", required=True, type=str, help="")
    parser.add_argument("--num_steps", required=True, type=int, help="")
    parser.add_argument("--benchmark_size", default=10, type=int, help="")
    parser.add_argument('--benchmark_exclude', metavar='N', type=int, nargs='+', help='')
    parser.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                        help="Choose platform to log results. NoPlatform means no logging.")
    parser.add_argument("--device", default=0, type=int, help="Device id to use.")
    args = parser.parse_args()

    last_reported_iter = args.num_steps-1

    benchmark_samples = np.arange(args.benchmark_size)
    if args.benchmark_exclude is not None:
        mask = np.isin(benchmark_samples, args.benchmark_exclude, invert=True)
        benchmark_samples = benchmark_samples[mask]
        exc_str = [str(num) for num in args.benchmark_exclude]
        exc_str = ','.join(exc_str)

    # Wait for all evaluation to finish - then read them
    last_eval_files = [os.path.join(args.benchmark_dir, f'{i:04d}', f'eval_{last_reported_iter:09d}.npy') for i in benchmark_samples]  #  FIXME -1
    last_eval_files_contd = [os.path.join(args.benchmark_dir, f'{i:04d}_contd', f'eval_{last_reported_iter:09d}.npy') for i in benchmark_samples]  # FIXME -1
    print(last_eval_files)

    shutil.copyfile(os.path.join(os.path.dirname(last_eval_files[0]), 'args.json'), os.path.join(args.benchmark_dir, 'args.json'))

    while not np.logical_or(np.array([os.path.exists(f) for f in last_eval_files]),np.array([os.path.exists(f) for f in last_eval_files_contd])).all():
        print('Waiting for all evaluations to finish. DO NOT KILL.')
        print('Missing files: ' + str([f for f in last_eval_files if not os.path.exists(f)]))
        time.sleep(30)


    # Open a clearml task
    with open(os.path.join(args.benchmark_dir, 'args.json'), 'r') as fr:
        orig_args = json.load(fr)
    train_platform_type = eval(args.train_platform_type)
    task_name = args.benchmark_dir + f'-benchmark-{args.num_steps}steps-{args.benchmark_size}samples'
    if args.benchmark_exclude:
        task_name += f'_exc{exc_str}'
    train_platform = train_platform_type(task_name)
    train_platform.report_args(orig_args, name='Args')

    # add reports from previous iterations, if exist
    all_eval_files = glob.glob(os.path.join(args.benchmark_dir, '*', f'eval_*.npy'))
    eval_dicts = {}
    for eval_file in all_eval_files:
        iter = re.findall('.*eval_(\d+).*.npy', eval_file)
        assert len(iter) == 1
        iter = int(iter[0])
        try:
            sample_i = re.findall('.*\/(\d\d\d\d)\/.*', eval_file)[0]
        except:
            sample_i = re.findall('.*\/(\d\d\d\d)_contd\/.*', eval_file)[0]
        vals = np.load(eval_file, allow_pickle=True)[None][0]
        if sample_i not in eval_dicts.keys():
            eval_dicts[sample_i] = dict()
        eval_dicts[sample_i][iter] = vals

    # Report Results
    for sample_i in benchmark_samples:
        for iter in eval_dicts[f'{sample_i:04d}']:
            for win_size in eval_dicts[f'{sample_i:04d}'][iter]:
                for m, vals in eval_dicts[f'{sample_i:04d}'][iter][win_size].items():
                    train_platform.report_scalar(name=f'sample{sample_i:04d}', value=vals['mean'], iteration=iter,
                                                 group_name=f'Eval_window{win_size}/{m}')

    log_file = os.path.join(args.benchmark_dir, f'eval.log')
    win_sizes = list(eval_dicts['0000'][last_reported_iter].keys())

    metric_names = eval_dicts['0000'][last_reported_iter][win_sizes[0]].keys()
    mean_dict = dict()

    for iter in eval_dicts['0000'].keys():
        mean_dict[iter] = dict()
        for win_size in win_sizes:
            mean_dict[iter][win_size] = dict()
            for m in metric_names:
                mean_dict[iter][win_size][m] = dict()
                mean_dict[iter][win_size][m]['mean'] = \
                    np.mean([eval_dicts[f'{sample_i:04d}'][iter][win_size][m]['mean'] for sample_i in benchmark_samples])
                mean_dict[iter][win_size][m]['std'] = \
                    np.std([eval_dicts[f'{sample_i:04d}'][iter][win_size][m]['mean'] for sample_i in benchmark_samples])
    print(mean_dict[last_reported_iter])

    np.save(log_file.replace('.log', '.npy'), mean_dict)
    with open(log_file, 'w') as fw:
        fw.write(str(mean_dict))
    for iter in eval_dicts['0000'].keys():
        for win_size in win_sizes:
            for m, vals in mean_dict[iter][win_size].items():
                train_platform.report_scalar(name=f'MEAN', value=vals['mean'], iteration=iter, group_name=f'Eval_window{win_size}/{m}')

    train_platform.close()
