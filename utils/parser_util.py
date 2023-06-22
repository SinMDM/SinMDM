from argparse import ArgumentParser
import argparse
import os
import json
import sys


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_single_motion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion', 'single_motion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), f'Arguments json file {args_path} was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compatability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='qna',
                       choices=['trans_enc', 'trans_dec', 'gru', 'unet', 'qna'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=128, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")

    # UNET options
    group.add_argument("--image_size", default=64, type=int, help="")
    group.add_argument("--num_channels", default=128, type=int, help="")
    group.add_argument("--num_res_blocks", default=2, type=int, help="")
    group.add_argument("--num_heads", default=4, type=int, help="")
    group.add_argument("--num_heads_upsample", default=-1, type=int, help="")
    group.add_argument("--num_head_channels", default=-1, type=int, help="")
    group.add_argument("--attention_resolutions", default="16,8", type=str, help="")
    group.add_argument("--channel_mult", default="", type=str, help="")
    group.add_argument("--learn_sigma", action='store_true', help="")
    group.add_argument("--dropout", default=0.0, type=float, help="")
    group.add_argument("--class_cond", action='store_true', help="")
    group.add_argument("--use_checkpoint", action='store_true', help="")
    group.add_argument("--use_scale_shift_norm", action='store_true', help="")
    group.add_argument("--resblock_updown", action='store_true', help="")
    group.add_argument("--use_fp16", action='store_true', help="")
    group.add_argument("--use_new_attention_order", action='store_true', help="")
    group.add_argument("--conv_1d", action='store_true', help="")
    group.add_argument("--padding_mode", default='replicate', choices=['zeros', 'reflect', 'replicate', 'circular'], type=str,
                       help="Padding mode during convolution. One of ['zeros', 'reflect', 'replicate', 'circular'].")
    group.add_argument("--padding", default=1, type=int, help="")
    group.add_argument("--lr_method", default=None, type=str, help="")
    group.add_argument("--lr_step", default=None, type=int, help="")
    group.add_argument("--lr_gamma", default=None, type=float, help="")
    group.add_argument("--use_attention", action='store_true', help="")
    group.add_argument("--use_qna", action='store_true', help="")

    # QnA Options:
    group.add_argument("--head_dim", default=32, type=int, help="")
    group.add_argument("--num_downsample", default=0, type=int, help="")
    group.add_argument("--drop_path", default=0.5, type=float, help="")
    group.add_argument("--use_diffusion_query", action='store_true', help="")
    group.add_argument("--kernel_size", default=7, type=int, help="")
    group.add_argument("--use_global_pe", action="store_true", help="")


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'mixamo', 'bvh_general'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--repr", default='6d', choices=['quat', '6d'],
                       type=str,
                       help="Motion representation (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--num_joints", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")



def add_single_motion_options(parser):
    group = parser.add_argument_group(('single_motion'))
    group.add_argument("--crop_ratio", default=3.6, type=float, help=".")
    group.add_argument("--sin_path", type=str, help=".")



def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--warmup_steps", default=1000, type=int, help="Number of warmup steps")
    group.add_argument("--weight_decay", default=0.02, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=1, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=500, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=10_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=20_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=4, type=int,
                       help="Number of samples to sample while generating")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=None, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")

def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_betweening', choices=['in_betweening', 'expansion', 'upper_body',  'lower_body', 'harmonization'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_betweening - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) expansion - prefix and suffix are generated,"
                            "middle motion is taken from input motion.\n"
                            "(3) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.\n"
                            "(4) lower_body - upper body joints taken from input motion, "
                            "lower body is generated.\n"
                            "(5) harmonization - low frequencies are taken from the reference motion,"
                            "high frequencies are generated"
                       )
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_betweening editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_betweening editing - Defines the start of input suffix (ratio from all frames).")
    group.add_argument("--prefix_length", default=0.15, type=float,
                       help="For motion expansion - Defines the length of generated prefix (ratio from all frames of input).")
    group.add_argument("--suffix_length", default=0.15, type=float,
                       help="For motion expansion - Defines the length of generated suffix (ratio from all frames of input).")
    group.add_argument("--ref_motion", type=str, default="", required=False,
                       help="Refenrence motion to be used for editing.")
    group.add_argument("--num_frames", type=int, default=-1, required=False)


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_special", default='none', choices=['none', 'self', 'other'], type=str,
                       help='evaluation of extreme case, for debug only.')



def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def train_sin_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_single_motion_options(parser)
    args = sys.argv[1:] + ['--unconstrained', '--cond_mask_prob', '0']  # a single image network is unconstrained (for now)
    return parser.parse_args(args)


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    return parse_and_load_from_model(parser)


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)