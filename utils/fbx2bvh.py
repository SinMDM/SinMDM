"""
This code is a variation of https://github.com/rubenvillegas/cvpr2018nkn/blob/master/datasets/fbx2bvh.py
"""
from glob import glob
import os
import os.path as osp

import bpy

print('start')
in_dir = '<path to folder containing fbx file>'
out_dir = in_dir + '/fbx2bvh'  # in_dir.replace('fbx', 'bvh')
fbx_files = glob(osp.join(in_dir, '*.fbx'))
for idx, in_file in enumerate(fbx_files):
    print(in_file)
    in_file_no_path = osp.split(in_file)[1]
    motion_name = osp.splitext(in_file_no_path)[0]
    rel_in_file = osp.relpath(in_file, in_dir)
    rel_out_file = osp.join(osp.split(rel_in_file)[0], '{}'.format(motion_name), '{}.bvh'.format(motion_name))
    rel_dir = osp.split(rel_out_file)[0]
    out_file = osp.join(out_dir, rel_out_file)

    os.makedirs(osp.join(out_dir, rel_dir), exist_ok=True)

    bpy.ops.import_scene.fbx(filepath=in_file)

    action = bpy.data.actions[-1]
    assert action.frame_range[0] < 9999 and action.frame_range[1] > -9999  # checking because of Kfir's code
    bpy.ops.export_anim.bvh(filepath=out_file,
                            frame_start=action.frame_range[0],
                            frame_end=action.frame_range[1], root_transform_only=True)
    bpy.data.actions.remove(bpy.data.actions[-1])

    print('{} processed. #{} of {}'.format(out_file, idx, len(fbx_files)))
