##############################
#
# https://github.com/PeizhuoLi/ganimator/blob/main/models/kinematics.py
#
##############################
import torch
import torch.nn.functional as F
from Motion.transforms import quat2mat, repr6d2mat, euler2mat


class ForwardKinematics:
    def __init__(self, parents, offsets=None):
        self.parents = parents
        if offsets is not None and len(offsets.shape) == 2:
            offsets = offsets.unsqueeze(0)
        self.offsets = offsets

    def forward(self, rots, offsets=None, global_pos=None):
        """
        Forward Kinematics: returns a per-bone transformation
        @param rots: local joint rotations (batch_size, bone_num, 3, 3)
        @param offsets: (batch_size, bone_num, 3) or None
        @param global_pos: global_position: (batch_size, 3) or keep it as in offsets (default)
        @return: (batch_szie, bone_num, 3, 4)
        """
        rots = rots.clone()
        if offsets is None:
            offsets = self.offsets.to(rots.device)
        if global_pos is None:
            global_pos = offsets[:, 0]

        pos = torch.zeros((rots.shape[0], rots.shape[1], 3), device=rots.device)
        rest_pos = torch.zeros_like(pos)
        res = torch.zeros((rots.shape[0], rots.shape[1], 3, 4), device=rots.device)

        pos[:, 0] = global_pos
        rest_pos[:, 0] = offsets[:, 0]

        for i, p in enumerate(self.parents):
            if i != 0:
                rots[:, i] = torch.matmul(rots[:, p], rots[:, i])
                pos[:, i] = torch.matmul(rots[:, p], offsets[:, i].unsqueeze(-1)).squeeze(-1) + pos[:, p]
                rest_pos[:, i] = rest_pos[:, p] + offsets[:, i]

            res[:, i, :3, :3] = rots[:, i]
            res[:, i, :, 3] = torch.matmul(rots[:, i], -rest_pos[:, i].unsqueeze(-1)).squeeze(-1) + pos[:, i]

        return res

    def accumulate(self, local_rots):
        """
        Get global joint rotation from local rotations
        @param local_rots: (batch_size, n_bone, 3, 3)
        @return: global_rotations
        """
        res = torch.empty_like(local_rots)
        for i, p in enumerate(self.parents):
            if i == 0:
                res[:, i] = local_rots[:, i]
            else:
                res[:, i] = torch.matmul(res[:, p], local_rots[:, i])
        return res

    def unaccumulate(self, global_rots):
        """
        Get local joint rotation from global rotations
        @param global_rots: (batch_size, n_bone, 3, 3)
        @return: local_rotations
        """
        res = torch.empty_like(global_rots)
        inv = torch.empty_like(global_rots)

        for i, p in enumerate(self.parents):
            if i == 0:
                inv[:, i] = global_rots[:, i].transpose(-2, -1)
                res[:, i] = global_rots[:, i]
                continue
            res[:, i] = torch.matmul(inv[:, p], global_rots[:, i])
            inv[:, i] = torch.matmul(res[:, i].transpose(-2, -1), inv[:, p])

        return res


class ForwardKinematicsJoint:
    def __init__(self, parents, offset):
        self.parents = parents
        self.offset = offset

    '''
        rotation should have shape batch_size * Joint_num * (3/4) * Time
        position should have shape batch_size * 3 * Time
        offset should have shape batch_size * Joint_num * 3
        output have shape batch_size * Time * Joint_num * 3
    '''

    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset=None,
                world=True):
        '''
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        position = position.permute(0, 2, 1)
        '''
        if rotation.shape[-1] == 6:
            transform = repr6d2mat(rotation)
        elif rotation.shape[-1] == 4:
            rotation = F.normalize(rotation, dim=-1)
            transform = quat2mat(rotation)
        elif rotation.shape[-1] == 3:
            transform = euler2mat(rotation)
        else:
            raise Exception('Only accept quaternion rotation input')
        result = torch.empty(transform.shape[:-2] + (3,), device=position.device)

        if offset is None:
            offset = self.offset
        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.parents):
            if pi == -1:
                assert i == 0
                continue

            result[..., i, :] = torch.matmul(transform[..., pi, :, :], offset[..., i, :, :]).squeeze(-1)
            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :].clone(), transform[..., i, :, :].clone())
            if world: result[..., i, :] += result[..., pi, :]
        return result


class InverseKinematicsJoint:
    def __init__(self, rotations: torch.Tensor, positions: torch.Tensor, offset, parents, constrains):
        self.rotations = rotations.detach().clone()
        self.rotations.requires_grad_(True)
        self.position = positions.detach().clone()
        self.position.requires_grad_(True)

        self.parents = parents
        self.offset = offset
        self.constrains = constrains

        self.optimizer = torch.optim.Adam([self.position, self.rotations], lr=1e-3, betas=(0.9, 0.999))
        self.criteria = torch.nn.MSELoss()

        self.fk = ForwardKinematicsJoint(parents, offset)

        self.glb = None

    def step(self):
        self.optimizer.zero_grad()
        glb = self.fk.forward(self.rotations, self.position)
        loss = self.criteria(glb, self.constrains)
        loss.backward()
        self.optimizer.step()
        self.glb = glb
        return loss.item()


class InverseKinematicsJoint2:
    def __init__(self, rotations: torch.Tensor, positions: torch.Tensor, offset, parents, constrains, cid,
                 lambda_rec_rot=1., lambda_rec_pos=1., use_velo=False):
        self.use_velo = use_velo
        self.rotations_ori = rotations.detach().clone()
        self.rotations = rotations.detach().clone()
        self.rotations.requires_grad_(True)
        self.position_ori = positions.detach().clone()
        self.position = positions.detach().clone()
        if self.use_velo:
            self.position[1:] = self.position[1:] - self.position[:-1]
        self.position.requires_grad_(True)

        self.parents = parents
        self.offset = offset
        self.constrains = constrains.detach().clone()
        self.cid = cid

        self.lambda_rec_rot = lambda_rec_rot
        self.lambda_rec_pos = lambda_rec_pos

        self.optimizer = torch.optim.Adam([self.position, self.rotations], lr=1e-3, betas=(0.9, 0.999))
        self.criteria = torch.nn.MSELoss()

        self.fk = ForwardKinematicsJoint(parents, offset)

        self.glb = None

    def step(self):
        self.optimizer.zero_grad()
        if self.use_velo:
            position = torch.cumsum(self.position, dim=0)
        else:
            position = self.position
        glb = self.fk.forward(self.rotations, position)
        self.constrain_loss = self.criteria(glb[:, self.cid], self.constrains)
        self.rec_loss_rot = self.criteria(self.rotations, self.rotations_ori)
        self.rec_loss_pos = self.criteria(self.position, self.position_ori)
        loss = self.constrain_loss + self.rec_loss_rot * self.lambda_rec_rot + self.rec_loss_pos * self.lambda_rec_pos
        loss.backward()
        self.optimizer.step()
        self.glb = glb
        return loss.item()

    def get_position(self):
        if self.use_velo:
            position = torch.cumsum(self.position.detach(), dim=0)
        else:
            position = self.position.detach()
        return position
