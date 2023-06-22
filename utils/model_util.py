from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from models.mdm_qnanet import QnAMDM
from models.mdm_unet import MDM_UNetModel


def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0


def create_model_and_diffusion(args, data, num_joints = None):
    motion_args = get_model_args(args, data, num_joints)
    if args.arch == 'unet':
        model = create_motion_unet(
            motion_args,
            args.image_size,
            args.num_channels,
            args.num_res_blocks,
            channel_mult=args.channel_mult,
            learn_sigma=args.learn_sigma,
            class_cond=args.class_cond,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=args.num_heads_upsample,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            resblock_updown=args.resblock_updown,
            use_fp16=args.use_fp16,
            use_new_attention_order=args.use_new_attention_order,
            conv_1d=args.conv_1d,
            padding_mode=args.padding_mode,
            padding=args.padding,
            use_attention=args.use_attention,
            use_qna=args.use_qna,
            kernel_size=args.kernel_size,
        )
    elif args.arch == 'qna':
        qna_args = dict(
            njoints=motion_args["njoints"],
            nfeats=motion_args["nfeats"],
            data_rep=motion_args["data_rep"],
            latent_dim=args.latent_dim,
            num_layers=args.layers,
            dataset=args.dataset,
            arch=args.arch,
            num_downsample=args.num_downsample,
            drop_path=args.drop_path,
            diffusion_query=args.use_diffusion_query,
            head_dim=args.head_dim,
            kernel_size=args.kernel_size,
            use_global_pe=args.use_global_pe
        )
        print(f"Choosen params: {qna_args}")
        model = QnAMDM(**qna_args)
    else:
        raise ValueError("Architecture specified not supported: %s" % (args.arch,))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data, num_joints):
    # default args
    action_emb = 'tensor'
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset == 'humanml':
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    if data is not None and hasattr(data, 'dataset') and  hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'mixamo':
        data_rep = 'mixamo_vec'
        if data is not None:
            njoints = data.raw_motion.shape[1]
        elif num_joints is not None:
            njoints = num_joints
        else:
            njoints = int(args.num_joints) if args.num_joints else 174
        nfeats = 1

    elif args.dataset == 'bvh_general':
        data_rep = 'bvh_general_vec'
        if data is not None:
            njoints = data.raw_motion.shape[1]
        elif num_joints is not None:
            njoints = num_joints
        else:
            assert args.num_joints
            njoints = int(args.num_joints)
        nfeats = 9 if args.repr == '6d' else 7
        njoints = njoints * nfeats
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'dataset': args.dataset}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        dataset=args.dataset,
    )


def create_motion_unet(
    motion_args,
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    conv_1d=False,
    padding_mode='zeros',
    padding=1,
    use_attention=False,
    use_qna=False,
    kernel_size=3,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if motion_args['dataset'] in ['humanml', 'mixamo', 'bvh_general']:
        ch = motion_args['njoints'] * motion_args['nfeats']
    else:
        raise 'dataset is not supported yet.'

    return MDM_UNetModel(
        motion_args=motion_args,
        image_size=image_size,
        in_channels=ch,
        model_channels=num_channels,
        out_channels=ch,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        dims=1 if conv_1d else 2,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        padding_mode=padding_mode,
        padding=padding,
        use_attention=use_attention,
        use_qna=use_qna,
        kernel_size=kernel_size,
    )