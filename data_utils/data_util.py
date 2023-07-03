import numpy as np
import torch
from data_utils.mixamo.motion import MotionData
from Motion.transforms import quat2repr6d
from Motion import BVH

def load_sin_motion(args):
    motion_data = None
    suffix = args.sin_path.lower()[-4:]
    assert suffix in ['.npy', '.bvh']
    if args.dataset == 'humanml':
        assert suffix == '.npy'
        try:
            motion = np.load(args.sin_path)  # only motion npy
            if len(motion.shape) == 2:
                motion = np.transpose(motion)
                motion = np.expand_dims(motion, axis=1)

        except:
            motion = np.array(np.load(args.sin_path, allow_pickle=True)[None][0]['motion_raw'][0])  # benchmark npy
        motion = torch.from_numpy(motion)
        motion = motion.permute(1, 0, 2)  # n_feats x n_joints x n_frames   ==> n_joints x n_feats x n_frames
        motion = motion.to(torch.float32)  # align with network dtype
    elif args.dataset == 'mixamo':  # bvh
        assert suffix == '.bvh'
        # 174 - 24 joint rotations (6d) + 3 root translation + 6*4 foot contact labels + 3 padding
        repr = 'repr6d' if args.repr == '6d' else 'quat'
        motion_data = MotionData(args.sin_path, padding=True, use_velo=True,
                                 repr=repr, contact=True, keep_y_pos=True,
                                 joint_reduction=True)
        _, raw_motion_joints, raw_motion_frames = motion_data.raw_motion.shape
        motion = motion_data.raw_motion.squeeze()
    else:
        assert args.dataset == 'bvh_general' and suffix == '.bvh'
        anim, _, _ = BVH.load(args.sin_path)
        if args.repr == '6d':
            repr_6d = quat2repr6d(torch.tensor(anim.rotations.qs))
            motion = np.concatenate([anim.positions, repr_6d], axis=2)
        else:
            motion = np.concatenate([anim.positions, anim.rotations.qs], axis=2)
        motion = torch.from_numpy(motion)
        motion = motion.permute(1, 2, 0)  # n_frames x n_joints x n_feats  ==> n_joints x n_feats x n_frames
        motion = motion.to(torch.float32)  # align with network dtype

    motion = motion.to(args.device)
    return motion, motion_data

