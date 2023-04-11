##############################
#
# https://github.com/PeizhuoLi/ganimator/blob/main/evaluations/patched_nn.py
#
##############################
import torch
import numpy as np
from data_utils.mixamo.bvh.bvh_parser import BVH_file
from data_utils.mixamo.bvh.bvh_writer import WriterWrapper
from Motion.transforms import repr6d2quat
from ganimator_eval_kernel import prepare_group_cost
from ganimator_eval_kernel import nn_dp as nn_dp_kernel


def nn_dp_fast(group_cost, tmin):
    L = group_cost.shape[0]
    L_target = group_cost.shape[-1]
    G = np.zeros((L + 5, ), dtype=np.float64)
    G.fill(np.inf)
    E = np.zeros(G.shape, dtype=np.int32)
    F = np.zeros_like(E)
    label = np.zeros(L, dtype=np.int32)

    nn_dp_kernel(G, E, F, group_cost, tmin, L, L_target)

    lengths = []
    seps = []
    p = L
    while p > 0:
        label[F[p]:p] = E[p] + np.arange(p - F[p])
        seps.append((E[p], E[p] + p - F[p]))
        lengths.append(p - F[p])
        p = F[p]

    return G[L], label


def group_cost_from_file(src_file, tgt_files, use_pos=False):
    if isinstance(src_file, torch.Tensor):
        src_pos = src_file
    else:
        src_bvh = BVH_file(src_file, joint_reduction=False, no_scale=True)
        src_pos = src_bvh.local_pos() if use_pos else src_bvh.to_tensor(repr='repr6d', rot_only=True)
        src_pos = src_pos.reshape(src_pos.shape[0], -1)

    if isinstance(tgt_files, torch.Tensor):
        tgt_pos = tgt_files
    else:
        if not isinstance(tgt_files, list):
            tgt_files = [tgt_files]
        tgt_poses = []
        for tgt_file in tgt_files:
            tgt_bvh = BVH_file(tgt_file, joint_reduction=False, no_scale=True)
            tgt_pos = tgt_bvh.local_pos() if use_pos else tgt_bvh.to_tensor(repr='repr6d', rot_only=True)
            tgt_poses.append(tgt_pos.reshape(tgt_pos.shape[0], -1))
        tgt_pos = torch.cat(tgt_poses, dim=0)

    src_pos = src_pos.unsqueeze(1)
    tgt_pos = tgt_pos.unsqueeze(0)
    cost = torch.norm(src_pos - tgt_pos, dim=2)

    L = cost.shape[0]
    L_target = cost.shape[1]
    group_cost = np.zeros((L, L + 1, L_target))
    group_cost.fill(np.inf)
    for i in range(L):
        group_cost[i, i] = 0

    prepare_group_cost(group_cost, cost)
    return group_cost


def patched_nn_main(src_file, tgt_files, tmin=30, use_pos=False, out_file=None):
    group_cost = group_cost_from_file(src_file, tgt_files, use_pos)
    val, label = nn_dp_fast(group_cost, tmin)

    if out_file is not None:
        src_bvh = BVH_file(src_file, joint_reduction=False, no_scale=True)
        src_tensor = src_bvh.to_tensor(repr='repr6d')

        writer = WriterWrapper(src_bvh.skeleton.parent, src_bvh.skeleton.offsets)
        tgt_tensors = []
        if not isinstance(tgt_files, list):
            tgt_files = [tgt_files]
        for tgt_file in tgt_files:
            tgt_bvh = BVH_file(tgt_file, joint_reduction=False, no_scale=True)
            tgt_tensor = tgt_bvh.to_tensor(repr='repr6d')
            tgt_tensors.append(tgt_tensor)

        tgt_tensor = torch.cat(tgt_tensors, dim=0)

        res = tgt_tensor[label]
        pos = src_tensor[..., -3:]
        rot = res[:, :-3]
        rot = rot.reshape(rot.shape[0], -1, 6)
        rot = repr6d2quat(rot)
        writer.write(out_file, rot, pos, names=src_bvh.skeleton.names)

    return val / label.shape[0]
