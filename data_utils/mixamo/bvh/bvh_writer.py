##############################
#
# https://github.com/PeizhuoLi/ganimator/blob/main/bvh/bvh_writer.py
#
##############################
import numpy as np
import torch
from Motion.transforms import quat2euler, repr6d2quat


# rotation with shape frame * J * 3
def write_bvh(parent, offset, rotation, position, names, frametime, order, path, endsite=None):
    file = open(path, 'w')
    frame = rotation.shape[0]
    joint_num = rotation.shape[1]
    order = order.upper()

    file_string = 'HIERARCHY\n'

    seq = []

    def write_static(idx, prefix):
        nonlocal parent, offset, rotation, names, order, endsite, file_string, seq
        seq.append(idx)
        if idx == 0:
            name_label = 'ROOT ' + names[idx]
            channel_label = 'CHANNELS 6 Xposition Yposition Zposition {}rotation {}rotation {}rotation'.format(*order)
        else:
            name_label = 'JOINT ' + names[idx]
            channel_label = 'CHANNELS 3 {}rotation {}rotation {}rotation'.format(*order)
        offset_label = 'OFFSET %.6f %.6f %.6f' % (offset[idx][0], offset[idx][1], offset[idx][2])

        file_string += prefix + name_label + '\n'
        file_string += prefix + '{\n'
        file_string += prefix + '\t' + offset_label + '\n'
        file_string += prefix + '\t' + channel_label + '\n'

        has_child = False
        for y in range(idx+1, rotation.shape[1]):
            if parent[y] == idx:
                has_child = True
                write_static(y, prefix + '\t')
        if not has_child:
            file_string += prefix + '\t' + 'End Site\n'
            file_string += prefix + '\t' + '{\n'
            file_string += prefix + '\t\t' + 'OFFSET 0 0 0\n'
            file_string += prefix + '\t' + '}\n'

        file_string += prefix + '}\n'

    write_static(0, '')

    file_string += 'MOTION\n' + 'Frames: {}\n'.format(frame) + 'Frame Time: %.8f\n' % frametime
    for i in range(frame):
        file_string += '%.6f %.6f %.6f ' % (position[i][0], position[i][1], position[i][2])
        for j in range(joint_num):
            idx = seq[j]
            file_string += '%.6f %.6f %.6f ' % (rotation[i][idx][0], rotation[i][idx][1], rotation[i][idx][2])
        file_string += '\n'

    file.write(file_string)
    return file_string


class WriterWrapper:
    def __init__(self, parents, offset=None):
        self.parents = parents
        self.offset = offset

    def write(self, filename, rot, pos, offset=None, names=None, repr='quat'):
        """
        Write animation to bvh file
        :param filename:
        :param rot: Quaternion as (w, x, y, z)
        :param pos:
        :param offset:
        :return:
        """
        if repr not in ['euler', 'quat', 'quaternion', 'repr6d']:
            raise Exception('Unknown rotation representation')
        if offset is None:
            offset = self.offset
        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(offset)
        n_bone = offset.shape[0]

        if repr == 'repr6d':
            rot = rot.reshape(rot.shape[0], -1, 6)
            rot = repr6d2quat(rot)
        if repr == 'repr6d' or repr == 'quat' or repr == 'quaternion':
            rot = rot.reshape(rot.shape[0], -1, 4)
            rot /= rot.norm(dim=-1, keepdim=True) ** 0.5
            euler = quat2euler(rot, order='xyz')
            rot = euler

        if names is None:
            names = ['%02d' % i for i in range(n_bone)]
        write_bvh(self.parents, offset, rot, pos, names, 1, 'xyz', filename)
