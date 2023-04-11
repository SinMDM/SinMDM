##############################
#
# https://github.com/PeizhuoLi/ganimator/blob/main/bvh/bvh_parser.py
#
##############################
import torch
import data_utils.mixamo.bvh.bvh_io as bvh_io
import numpy as np
from data_utils.mixamo.bvh.Quaternions import Quaternions
from data_utils.mixamo.bvh.skeleton_database import SkeletonDatabase
from data_utils.mixamo.models.kinematics import ForwardKinematicsJoint
from Motion.transforms import quat2repr6d
from data_utils.mixamo.models.contact import foot_contact
from data_utils.mixamo.bvh.bvh_writer import WriterWrapper


class Skeleton:
    def __init__(self, names, parent, offsets, joint_reduction=True):
        self._names = names
        self.original_parent = parent
        self._offsets = offsets
        self._parent = None
        self._ee_id = None
        self.contact_names = []

        for i, name in enumerate(self._names):
            if ':' in name:
                self._names[i] = name[name.find(':')+1:]

        if joint_reduction:
            self.skeleton_type, match_num = SkeletonDatabase.match(names)
            corps_names = SkeletonDatabase.corps_names[self.skeleton_type]
            self.contact_names = SkeletonDatabase.contact_names[self.skeleton_type]
            self.contact_threshold = SkeletonDatabase.contact_thresholds[self.skeleton_type]

            self.contact_id = []
            for i in self.contact_names:
                self.contact_id.append(corps_names.index(i))
        else:
            self.skeleton_type = -1
            corps_names = self._names

        self.details = []    # joints that does not belong to the corps (we are not interested in them)
        for i, name in enumerate(self._names):
            if name not in corps_names: self.details.append(i)

        self.corps = []
        self.simplified_name = []
        self.simplify_map = {}
        self.inverse_simplify_map = {}

        # Repermute the skeleton id according to the databse
        for name in corps_names:
            for j in range(len(self._names)):
                if name in self._names[j]:
                    self.corps.append(j)
                    break
        if len(self.corps) != len(corps_names):
            for i in self.corps:
                print(self._names[i], end=' ')
            print(self.corps, self.skeleton_type, len(self.corps), sep='\n')
            raise Exception('Problem in this skeleton')

        self.joint_num_simplify = len(self.corps)
        for i, j in enumerate(self.corps):
            self.simplify_map[j] = i
            self.inverse_simplify_map[i] = j
            self.simplified_name.append(self._names[j])
        self.inverse_simplify_map[0] = -1
        for i in range(len(self._names)):
            if i in self.details:
                self.simplify_map[i] = -1

    @property
    def parent(self):
        if self._parent is None:
            self._parent = self.original_parent[self.corps].copy()
            for i in range(self._parent.shape[0]):
                if i >= 1: self._parent[i] = self.simplify_map[self._parent[i]]
            self._parent = tuple(self._parent)
        return self._parent

    @property
    def offsets(self):
        return torch.tensor(self._offsets[self.corps], dtype=torch.float)

    @property
    def names(self):
        return self.simplified_name

    @property
    def ee_id(self):
        raise Exception('Abaddoned')
        # if self._ee_id is None:
        #     self._ee_id = []
        #     for i in SkeletonDatabase.ee_names[self.skeleton_type]:
        #         self.ee_id._ee_id(corps_names[self.skeleton_type].index(i))


class BVH_file:
    def __init__(self, file_path, no_scale=False, requires_contact=False, joint_reduction=True):
        self.anim = bvh_io.load(file_path)
        self._names = self.anim.names
        self.frametime = self.anim.frametime
        self.skeleton = Skeleton(self.anim.names, self.anim.parent, self.anim.offsets, joint_reduction)

        # Downsample to 30 fps for our application
        if self.frametime < 0.0084:
            self.frametime *= 2
            self.anim.positions = self.anim.positions[::2]
            self.anim.rotations = self.anim.rotations[::2]
        if self.frametime < 0.017:
            self.frametime *= 2
            self.anim.positions = self.anim.positions[::2]
            self.anim.rotations = self.anim.rotations[::2]

        # Scale by 1/100 if it's raw exported bvh from blender
        if not no_scale and self.skeleton.offsets[0, 1] > 10:
            self.scale(1. / 100)
        # Scale by 0.14 for Xia's dataset
        if not no_scale and self.skeleton.skeleton_type == 3:
            self.scale(0.14)

        self.requires_contact = requires_contact

        if requires_contact:
            self.contact_names = self.skeleton.contact_names
        else:
            self.contact_names = []

        self.fk = ForwardKinematicsJoint(self.skeleton.parent, self.skeleton.offsets)
        self.writer = WriterWrapper(self.skeleton.parent, self.skeleton.offsets)
        if self.requires_contact:
            gl_pos = self.joint_position()
            self.contact_label = foot_contact(gl_pos[:, self.skeleton.contact_id],
                                              threshold=self.skeleton.contact_threshold)
            self.gl_pos = gl_pos

    def local_pos(self):
        gl_pos = self.joint_position()
        local_pos = gl_pos - gl_pos[:, 0:1, :]
        return local_pos[:, 1:]

    def scale(self, ratio):
        self.anim.offsets *= ratio
        self.anim.positions *= ratio

    def to_tensor(self, repr='euler', rot_only=False):
        if repr not in ['euler', 'quat', 'quaternion', 'repr6d']:
            raise Exception('Unknown rotation representation')
        positions = self.get_position()
        rotations = self.get_rotation(repr=repr)

        if rot_only:
            return rotations.reshape(rotations.shape[0], -1)

        if self.requires_contact:
            virtual_contact = torch.zeros_like(rotations[:, :len(self.skeleton.contact_id)])
            virtual_contact[..., 0] = self.contact_label
            rotations = torch.cat([rotations, virtual_contact], dim=1)

        rotations = rotations.reshape(rotations.shape[0], -1)
        return torch.cat((rotations, positions), dim=-1)

    def joint_position(self):
        positions = torch.tensor(self.anim.positions[:, 0, :], dtype=torch.float)
        rotations = self.anim.rotations[:, self.skeleton.corps, :]
        rotations = Quaternions.from_euler(np.radians(rotations)).qs
        rotations = torch.tensor(rotations, dtype=torch.float)
        j_loc = self.fk.forward(rotations, positions)
        return j_loc

    def get_rotation(self, repr='quat'):
        rotations = self.anim.rotations[:, self.skeleton.corps, :]
        if repr == 'quaternion' or repr == 'quat' or repr == 'repr6d':
            rotations = Quaternions.from_euler(np.radians(rotations)).qs
            rotations = torch.tensor(rotations, dtype=torch.float)
        if repr == 'repr6d':
            rotations = quat2repr6d(rotations)
        if repr == 'euler':
            rotations = torch.tensor(rotations, dtype=torch.float)
        return rotations

    def get_position(self):
        return torch.tensor(self.anim.positions[:, 0, :], dtype=torch.float)

    def dfs(self, x, vis, dist):
        fa = self.skeleton.parent
        vis[x] = 1
        for y in range(len(fa)):
            if (fa[y] == x or fa[x] == y) and vis[y] == 0:
                dist[y] = dist[x] + 1
                self.dfs(y, vis, dist)

    def get_neighbor(self, threshold, enforce_contact=False):
        fa = self.skeleton.parent
        neighbor_list = []
        for x in range(0, len(fa)):
            vis = [0 for _ in range(len(fa))]
            dist = [0 for _ in range(len(fa))]
            self.dfs(x, vis, dist)
            neighbor = []
            for j in range(0, len(fa)):
                if dist[j] <= threshold:
                    neighbor.append(j)
            neighbor_list.append(neighbor)

        contact_list = []
        if self.requires_contact:
            for i, p_id in enumerate(self.skeleton.contact_id):
                v_id = len(neighbor_list)
                neighbor_list[p_id].append(v_id)
                neighbor_list.append(neighbor_list[p_id])
                contact_list.append(v_id)

        root_neighbor = neighbor_list[0]
        id_root = len(neighbor_list)

        if enforce_contact:
            root_neighbor = root_neighbor + contact_list
            for j in contact_list:
                neighbor_list[j] = list(set(neighbor_list[j]))

        root_neighbor = list(set(root_neighbor))
        for j in root_neighbor:
            neighbor_list[j].append(id_root)
        root_neighbor.append(id_root)
        neighbor_list.append(root_neighbor)  # Neighbor for root position
        return neighbor_list
