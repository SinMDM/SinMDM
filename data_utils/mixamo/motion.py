##############################
#
# https://github.com/PeizhuoLi/ganimator/blob/main/dataset/motion.py
#
##############################
from os.path import join as pjoin
import numpy as np
import torch
import torch.nn.functional as F
from data_utils.mixamo.bvh.bvh_parser import BVH_file
from Motion.transforms import interpolate_6d


class MotionData:
    def __init__(self, filename, repr='quat', padding=False,
                 use_velo=False, no_scale=False, contact=False, keep_y_pos=False,
                 joint_reduction=True):
        self.bvh_file = BVH_file(filename, no_scale, requires_contact=contact,
                                 joint_reduction=joint_reduction)
        self.contact = contact
        self.filename = filename
        self.raw_motion = self.bvh_file.to_tensor(repr=repr)
        if repr == 'quat':
            self.n_rot = 4
        elif repr == 'repr6d':
            self.n_rot = 6
        elif repr == 'euler':
            self.n_rot = 3
        self.repr = repr
        self.use_velo = use_velo
        self.keep_y_pos = keep_y_pos

        self.writer = self.bvh_file.writer
        self.raw_motion = self.raw_motion.permute(1, 0)
        self.raw_motion.unsqueeze_(0)     # Shape = (1, n_channel, n_frames)
        if self.use_velo:
            self.velo_mask = [-3, -2, -1] if not keep_y_pos else [-3, -1]
            self.raw_motion[:, self.velo_mask, 1:] = self.raw_motion[:, self.velo_mask, 1:] - \
                                                     self.raw_motion[:, self.velo_mask, :-1]
            self.begin_pos = self.raw_motion[:, self.velo_mask, 0].clone()
            self.raw_motion[:, self.velo_mask, 0] = self.raw_motion[:, self.velo_mask, 1]


        if padding:
            self.n_pad = self.n_rot - 3 # pad position channels
            paddings = torch.zeros_like(self.raw_motion[:, :self.n_pad])
            self.raw_motion = torch.cat((self.raw_motion, paddings), dim=1)
        else:
            self.n_pad = 0

        self.raw_motion = torch.cat((self.raw_motion[:, :-3-self.n_pad], self.raw_motion[:, -3-self.n_pad:]), dim=1)

        if contact:
            self.n_contact = len(self.bvh_file.skeleton.contact_id)
        else:
            self.n_contact = 0

    def slerp(self, input, size):
        res = torch.empty(input.shape[:-1] + (size,), device=input.device, dtype=input.dtype)
        n_other = 3 + self.n_pad + self.n_contact * 6
        rot = input[..., :-n_other, :]
        other = input[..., -n_other:, :]

        rot_res = interpolate_6d(rot, size)
        other_res = F.interpolate(other, size=size, mode='linear', align_corners=True)
        res[..., :-n_other, :] = rot_res
        res[..., -n_other:, :] = other_res
        return res

    def sample(self, size=None, slerp=False, input=None):
        raw_motion = self.raw_motion
        if input is not None:
            raw_motion = input
        if size is None:
            return raw_motion
        else:
            if slerp:
                motion = self.slerp(raw_motion, size=size)
            else:
                motion = F.interpolate(raw_motion, size=size, mode='linear', align_corners=False)
            return motion

    @property
    def n_channels(self):
        return self.raw_motion.shape[1]

    def velo2pos(self, motion):
        if not self.use_velo:
            return motion
            # raise Exception('No need to call this function')
        res = motion.clone()
        mask = [i - self.n_pad for i in self.velo_mask]
        res[:, mask, 0] = self.begin_pos.to(motion.device)
        res[:, mask] = torch.cumsum(res[:, mask], dim=-1)
        return res

    def inject_contact(self, source, target):
        batch_size = source.shape[0]
        temporal_size = source.shape[-1]
        source = source.reshape(batch_size, -1, self.n_rot, temporal_size)
        target = target.clone().reshape(batch_size, -1, self.n_rot, temporal_size)
        target[:, -(self.n_contact+1):-1] = source[:, -(self.n_contact+1):-1]
        target = target.reshape(batch_size, -1, temporal_size)
        return target

    def parse(self, motion, keep_velo=False,):
        """
        :returns pos, rot, contact (if exists)
        """
        motion = motion.clone()

        input_ndims = motion.ndim
        if input_ndims < 4:
            motion = motion.unsqueeze(0)

        if self.n_pad:
            motion = motion[:, :, :-self.n_pad]
        if self.use_velo and not keep_velo:
            motion[:, :, self.velo_mask, 0] = self.begin_pos.to(motion.device)
            motion[:, :, self.velo_mask] = torch.cumsum(motion[:, :, self.velo_mask], dim=-1)
        motion = motion.squeeze(1).permute(0, 2, 1)
        pos = motion[..., -3:]
        rot = motion[..., :-3].reshape(motion.shape[0], motion.shape[1], -1, self.n_rot)
        if self.contact:
            contact = rot[..., -self.n_contact:, 0]
            rot = rot[..., :-self.n_contact, :]
        else:
            contact = None

        if input_ndims < 4:
            pos = pos.squeeze(0)
            rot = rot.squeeze(0)
            if contact is not None:
                contact = contact.squeeze(0)

        return pos, rot, contact


    def write(self, filename, motion):
        pos, rot, contact = self.parse(motion)
        if self.contact:
            np.save(filename + '.contact', contact.detach().cpu().numpy())
        self.writer.write(filename, rot, pos, names=self.bvh_file.skeleton.names, repr=self.repr)

    def __len__(self):
        return self.raw_motion.shape[-1]


def load_multiple_dataset(prefix, name_list, **kargs):
    with open(name_list, 'r') as f:
        names = [line.strip() for line in f.readlines()]
    datasets = []
    for f in names:
        kargs['filename'] = pjoin(prefix, f)
        datasets.append(MotionData(**kargs))
    return datasets
