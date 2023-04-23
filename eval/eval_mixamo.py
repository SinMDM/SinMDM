# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from data_utils.mixamo.motion import MotionData
from data_utils.mixamo.evaluations.patched_nn import patched_nn_main
from data_utils.mixamo.evaluations.perwindow_nn import perwindow_nn, coverage, avg_per_frame_dist
from tqdm import tqdm
from utils.fixseed import fixseed
import numpy as np
import torch
from utils.parser_util import generate_args, evaluation_parser
from utils.model_util import create_model_and_diffusion, load_model
from utils import dist_util
from data_utils.tensors import collate
import os


def mixamo_evaluate(args, model, diffusion):
    fixseed(args.seed)
    # Load GT
    if args.dataset == 'mixamo':
        motion_data = MotionData(args.sin_path, padding=True, use_velo=True,
                                 repr='repr6d', contact=True, keep_y_pos=True,
                                 joint_reduction=True)
        fps = int(round(1 / motion_data.bvh_file.frametime))
        assert fps == 30
        multiple_data = [motion_data]

        # Hardcoded according to Ganimator code:
        n_frames = len(motion_data) *2
        num_samples = 20
    else:
        raise NotImplementedError()

    slerp =0
    data = None

    is_using_data = False # todo: fix this hack. not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * num_samples
        _, model_kwargs = collate(collate_args)

    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
        model,
        (num_samples, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    all_samples = torch.tensor(sample).squeeze()

    all_samples = all_samples.permute(0, 2, 1)[..., :-6].detach().cpu()
    n_samples = all_samples.shape[0]

    global_variations = []
    local_variations = []
    coverages = []
    inter_div_local = []
    inter_div_dist = []
    intra_div_dist = []
    gt_intra_div_dist = []

    assert len(multiple_data) == 1
    motion_data = multiple_data[0]
    gt = motion_data.sample(size=len(motion_data), slerp=slerp).to(args.device)[0]
    gt = gt.permute(1, 0)[..., :-6].cpu()

    tmin = 15
    offsets = np.random.randint(n_frames-tmin, size=(n_samples,2))
    gt_offsets = np.random.randint(n_frames//2-1-tmin, size=(n_samples,2))

    if len(motion_data) > 1:
        print(f'Evaluating on sequence {0}...')
    else:
        print('Evaluating...')

    loop = tqdm(range(n_samples))
    for i in loop:
        global_variations.append(patched_nn_main(all_samples[i], gt))
        local_variations.append(perwindow_nn(all_samples[i], gt, tmin=tmin))
        inter_div_local.append(perwindow_nn(all_samples[i], all_samples[i-1], tmin=tmin))
        inter_div_dist.append(avg_per_frame_dist(all_samples[i], all_samples[i-1]))
        intra_div_dist.append(avg_per_frame_dist(all_samples[i, offsets[i,0]:offsets[i,0]+tmin], all_samples[i, offsets[i,1]:offsets[i,1]+tmin]))
        gt_intra_div_dist.append(avg_per_frame_dist(gt[gt_offsets[i,0]:gt_offsets[i,0]+tmin], gt[gt_offsets[i,1]:gt_offsets[i,1]+tmin]))
        coverages.append(coverage(all_samples[i], gt))
    loop.close()

    return {
        'coverage': {'mean': np.mean(coverages) * 100, 'std': np.std(coverages) * 100},
        'global_diversity': {'mean': np.mean(global_variations), 'std': np.std(global_variations)},
        'local_diversity': {'mean': np.mean(local_variations), 'std': np.std(local_variations)},
        'inter_diversity_local': {'mean': np.mean(inter_div_local), 'std': np.std(inter_div_local)},
        'inter_diversity_dist': {'mean': np.mean(inter_div_dist), 'std': np.std(inter_div_dist)},
        'intra_diversity_dist': {'mean': np.mean(intra_div_dist), 'std': np.std(intra_div_dist)},
        'gt_intra_diversity_dist': {'mean': np.mean(gt_intra_div_dist), 'std': np.std(gt_intra_div_dist)},
    }


if __name__ == "__main__":
    args = evaluation_parser()
    model, diffusion = create_model_and_diffusion(args, None)
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    log_file = os.path.join(os.path.dirname(args.model_path),
                            os.path.basename(args.model_path).replace('model', 'eval_').replace('.pt', '.log'))

    print(f'Will save to log file [{log_file}]')

    model.to(dist_util.dev())
    diffusion.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()  # disable random masking

    eval_dict = mixamo_evaluate(args, model, diffusion)
    print(eval_dict)
    with open(log_file, 'w') as fw:
        fw.write(str(eval_dict))
    np.save(log_file.replace('.log', '.npy'), eval_dict)
