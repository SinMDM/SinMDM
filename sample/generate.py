# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import copy
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model
from utils import dist_util
from data_utils.get_data import get_dataset_loader
from data_utils.humanml.scripts.motion_process import recover_from_ric
import data_utils.humanml.utils.paramUtil as paramUtil
from data_utils.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_utils.tensors import collate
from data_utils.mixamo.motion import MotionData
from Motion.transforms import repr6d2quat
from Motion import BVH
from Motion.Animation import positions_global as anim_pos
from Motion.Animation import Animation
from Motion.AnimationStructure import get_kinematic_chain
from Motion.Quaternions import Quaternions


def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    motion_data = None
    num_joints = None
    repr = 'repr6d' if args.repr == '6d' else 'quat'
    if args.dataset == 'mixamo':
        motion_data = MotionData(args.sin_path, padding=True, use_velo=True,
                                 repr=repr, contact=True, keep_y_pos=True,
                                 joint_reduction=True)
        fps = int(round(1 / motion_data.bvh_file.frametime))
        n_frames = motion_data.bvh_file.anim.shape[0]
        skeleton = get_kinematic_chain(motion_data.bvh_file.skeleton.parent)
        num_joints = motion_data.raw_motion.shape[1]
    elif args.dataset == 'bvh_general':
        sin_anim, joint_names, frametime = BVH.load(args.sin_path)
        fps = int(round(1 / frametime))
        skeleton = get_kinematic_chain(sin_anim.parents)
        n_frames = sin_anim.shape[0]
        num_joints = sin_anim.shape[1]
    else:
        assert args.dataset == 'humanml'
        fps = 20
    max_frames = 196 if args.dataset == 'humanml' else 60
    if args.motion_length is not None:
        n_frames = int(args.motion_length*fps)
    elif not args.dataset in ['mixamo', 'bvh_general']:
        n_frames = max_frames
    is_using_data = False # todo: fix this hack. not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger than default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    if args.dataset in ['humanml']:
        print('Loading dataset...')
        data = load_dataset(args, max_frames, n_frames)
    else:
        data = None
    total_num_samples = args.num_samples

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, motion_data, num_joints)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    diffusion.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()  # disable random masking

    model.requires_grad_(False)
    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []

    print(f'### Sampling')

    sample_fn = diffusion.p_sample_loop

    sample = sample_fn(
        model,
        (args.batch_size, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        n_joints = 22
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        skeleton = paramUtil.t2m_kinematic_chain
    # Recover XYZ *positions* from zoo vector representation
    elif model.data_rep in ['mixamo_vec', 'bvh_general_vec']:
        sample = sample.cpu().numpy()
        sample = sample.transpose(0, 3, 1, 2)  # n_samples x n_features x n_joints x n_frames  ==>   n_samples x n_frames x n_joints x n_features
        if args.dataset == 'mixamo':
            xyz_samples = np.zeros((args.num_samples, n_frames, 24, 3))  # shape it to match the output of anim_pos
        else:
            joint_features_length = 7 if args.repr=='quat' else 9
            assert model.njoints % joint_features_length == 0
            xyz_samples = np.zeros((args.num_samples, n_frames, int(model.njoints / joint_features_length), 3))  # shape it to match the output of anim_pos
        for i, one_sample in enumerate(sample):
            bvh_path = os.path.join(out_path, f'sample{i:02d}.bvh')
            if args.dataset == 'mixamo':
                motion_data.write(os.path.expanduser(bvh_path), torch.tensor(one_sample.transpose((2, 1, 0))))
                generated_motion = MotionData(os.path.expanduser(bvh_path), padding=True,
                                    use_velo=True, repr=repr, contact=True, keep_y_pos=True, joint_reduction=True)
                anim = Animation(rotations=Quaternions(generated_motion.bvh_file.get_rotation().numpy()),
                                 positions=generated_motion.bvh_file.anim.positions,
                                 orients=generated_motion.bvh_file.anim.orients,
                                 offsets=generated_motion.bvh_file.skeleton.offsets,
                                 parents=generated_motion.bvh_file.skeleton.parent)
            else:
                if args.repr == '6d':
                    one_sample = one_sample.reshape(n_frames, -1, joint_features_length)
                    quats = repr6d2quat(torch.tensor(one_sample[:, :, 3:])).numpy()
                else:
                    one_sample = one_sample.reshape(n_frames, -1, joint_features_length)
                    quats = one_sample[:, :, 3:]
                anim = Animation(rotations=Quaternions(quats), positions=one_sample[:, :, :3],
                                 orients=sin_anim.orients, offsets=sin_anim.offsets, parents=sin_anim.parents)
                BVH.save(os.path.expanduser(bvh_path), anim, joint_names, frametime, positions=True)  # "positions=True" is important for the dragon and does not harm the others
            xyz_samples[i] = anim_pos(anim)  # n_frames x n_joints x 3  =>
        sample = xyz_samples.transpose(0, 2, 3, 1)  # n_samples x n_frames x n_joints x 3  =>  n_samples x n_joints x 3 x n_frames

    rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec', 'mixamo_vec', 'bvh_general_vec'] else model.data_rep
    rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
    assert rot2xyz_pose_rep == 'xyz'

    if args.unconstrained:
        all_text += ['generated'] * args.num_samples
    else:
        text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
        all_text += model_kwargs['y'][text_key]

    if isinstance(sample, torch.Tensor):
        sample = sample.cpu().numpy()
    all_motions.append(sample)
    all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

    print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]


    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []
        caption = all_text[0 *args.batch_size + sample_i]
        length = all_lengths[0 *args.batch_size + sample_i]
        motion = all_motions[0 *args.batch_size + sample_i].transpose(2, 0, 1)[:length]  # n_joints x 3 x n_frames  ==>  n_frames x n_joints x 3
        save_file = sample_file_template.format(sample_i, 0 )
        print(sample_print_template.format(caption, sample_i, 0 , save_file))
        animation_save_path = os.path.join(out_path, save_file)
        motion_to_plot = copy.deepcopy(motion)
        if 'Breakdancing_Dragon' in args.sin_path:
            motion_to_plot = motion_to_plot[:, :, [0,2,1]] # replace y and z axes
        plot_3d_motion(animation_save_path, skeleton, motion_to_plot, dataset=args.dataset, title=caption, fps=fps)
        # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        rep_files.append(animation_save_path)

        sample_files, all_sample_save_path = save_multiple_samples(args, out_path,
                                               row_print_template, all_print_template, row_file_template, all_file_template,
                                               caption, num_samples_in_out_file, rep_files, sample_files, sample_i)
    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

    assert all_sample_save_path is not None
    return all_sample_save_path


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    all_sample_save_path = None
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # save several samples together
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files, all_sample_save_path


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template



def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
