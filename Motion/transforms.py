##############################
#
#https://github.com/PeizhuoLi/ganimator/blob/main/models/transforms.py
#
##############################
import numpy as np
import torch
import torch.nn.functional as F


def batch_mm(matrix, matrix_batch):
    """
    https://github.com/pytorch/pytorch/issues/14489#issuecomment-607730242
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)


def aa2quat(rots, form='wxyz', unified_orient=True):
    """
    Convert angle-axis representation to wxyz quaternion and to the half plan (w >= 0)
    @param rots: angle-axis rotations, (*, 3)
    @param form: quaternion format, either 'wxyz' or 'xyzw'
    @param unified_orient: Use unified orientation for quaternion (quaternion is dual cover of SO3)
    :return:
    """
    angles = rots.norm(dim=-1, keepdim=True)
    norm = angles.clone()
    norm[norm < 1e-8] = 1
    axis = rots / norm
    quats = torch.empty(rots.shape[:-1] + (4,), device=rots.device, dtype=rots.dtype)
    angles = angles * 0.5
    if form == 'wxyz':
        quats[..., 0] = torch.cos(angles.squeeze(-1))
        quats[..., 1:] = torch.sin(angles) * axis
    elif form == 'xyzw':
        quats[..., :3] = torch.sin(angles) * axis
        quats[..., 3] = torch.cos(angles.squeeze(-1))

    if unified_orient:
        idx = quats[..., 0] < 0
        quats[idx, :] *= -1

    return quats


def quat2aa(quats):
    """
    Convert wxyz quaternions to angle-axis representation
    :param quats:
    :return:
    """
    _cos = quats[..., 0]
    xyz = quats[..., 1:]
    _sin = xyz.norm(dim=-1)
    norm = _sin.clone()
    norm[norm < 1e-7] = 1
    axis = xyz / norm.unsqueeze(-1)
    angle = torch.atan2(_sin, _cos) * 2
    return axis * angle.unsqueeze(-1)


def quat2mat(quats: torch.Tensor):
    """
    Convert (w, x, y, z) quaternions to 3x3 rotation matrix
    :param quats: quaternions of shape (..., 4)
    :return:  rotation matrices of shape (..., 3, 3)
    """
    qw = quats[..., 0]
    qx = quats[..., 1]
    qy = quats[..., 2]
    qz = quats[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = torch.empty(quats.shape[:-1] + (3, 3), device=quats.device, dtype=quats.dtype)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m


def quat2euler(q, order='xyz', degrees=True):
    """
    Convert (w, x, y, z) quaternions to xyz euler angles. This is  used for bvh output.
    """
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]
    es = torch.empty(q0.shape + (3,), device=q.device, dtype=q.dtype)

    if order == 'xyz':
        es[..., 2] = torch.atan2(2 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        es[..., 1] = torch.asin((2 * (q1 * q3 + q0 * q2)).clip(-1, 1))
        es[..., 0] = torch.atan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
    else:
        raise NotImplementedError('Cannot convert to ordering %s' % order)

    if degrees:
        es = es * 180 / np.pi

    return es


def euler2mat(rots, order='xyz'):
    axis = {'x': torch.tensor((1, 0, 0), device=rots.device),
            'y': torch.tensor((0, 1, 0), device=rots.device),
            'z': torch.tensor((0, 0, 1), device=rots.device)}

    rots = rots / 180 * np.pi
    mats = []
    for i in range(3):
        aa = axis[order[i]] * rots[..., i].unsqueeze(-1)
        mats.append(aa2mat(aa))
    return mats[0] @ (mats[1] @ mats[2])


def aa2mat(rots):
    """
    Convert angle-axis representation to rotation matrix
    :param rots: angle-axis representation
    :return:
    """
    quat = aa2quat(rots)
    mat = quat2mat(quat)
    return mat


def mat2quat(R) -> torch.Tensor:
    '''
    https://github.com/duolu/pyrotation/blob/master/pyrotation/pyrotation.py
    Convert a rotation matrix to a unit quaternion.

    This uses the Shepperd’s method for numerical stability.
    '''

    # The rotation matrix must be orthonormal

    w2 = (1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2])
    x2 = (1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])
    y2 = (1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])
    z2 = (1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])

    yz = (R[..., 1, 2] + R[..., 2, 1])
    xz = (R[..., 2, 0] + R[..., 0, 2])
    xy = (R[..., 0, 1] + R[..., 1, 0])

    wx = (R[..., 2, 1] - R[..., 1, 2])
    wy = (R[..., 0, 2] - R[..., 2, 0])
    wz = (R[..., 1, 0] - R[..., 0, 1])

    w = torch.empty_like(x2)
    x = torch.empty_like(x2)
    y = torch.empty_like(x2)
    z = torch.empty_like(x2)

    flagA = (R[..., 2, 2] < 0) * (R[..., 0, 0] > R[..., 1, 1])
    flagB = (R[..., 2, 2] < 0) * (R[..., 0, 0] <= R[..., 1, 1])
    flagC = (R[..., 2, 2] >= 0) * (R[..., 0, 0] < -R[..., 1, 1])
    flagD = (R[..., 2, 2] >= 0) * (R[..., 0, 0] >= -R[..., 1, 1])

    x[flagA] = torch.sqrt(x2[flagA])
    w[flagA] = wx[flagA] / x[flagA]
    y[flagA] = xy[flagA] / x[flagA]
    z[flagA] = xz[flagA] / x[flagA]

    y[flagB] = torch.sqrt(y2[flagB])
    w[flagB] = wy[flagB] / y[flagB]
    x[flagB] = xy[flagB] / y[flagB]
    z[flagB] = yz[flagB] / y[flagB]

    z[flagC] = torch.sqrt(z2[flagC])
    w[flagC] = wz[flagC] / z[flagC]
    x[flagC] = xz[flagC] / z[flagC]
    y[flagC] = yz[flagC] / z[flagC]

    w[flagD] = torch.sqrt(w2[flagD])
    x[flagD] = wx[flagD] / w[flagD]
    y[flagD] = wy[flagD] / w[flagD]
    z[flagD] = wz[flagD] / w[flagD]

    # if R[..., 2, 2] < 0:
    #
    #     if R[..., 0, 0] > R[..., 1, 1]:
    #
    #         x = torch.sqrt(x2)
    #         w = wx / x
    #         y = xy / x
    #         z = xz / x
    #
    #     else:
    #
    #         y = torch.sqrt(y2)
    #         w = wy / y
    #         x = xy / y
    #         z = yz / y
    #
    # else:
    #
    #     if R[..., 0, 0] < -R[..., 1, 1]:
    #
    #         z = torch.sqrt(z2)
    #         w = wz / z
    #         x = xz / z
    #         y = yz / z
    #
    #     else:
    #
    #         w = torch.sqrt(w2)
    #         x = wx / w
    #         y = wy / w
    #         z = wz / w

    res = [w, x, y, z]
    res = [z.unsqueeze(-1) for z in res]

    return torch.cat(res, dim=-1) / 2


def quat2repr6d(quat):
    mat = quat2mat(quat)
    res = mat[..., :2, :]
    res = res.reshape(res.shape[:-2] + (6, ))
    return res


def repr6d2mat(repr):
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat


def repr6d2quat(repr) -> torch.Tensor:
    x = repr[..., :3]
    y = repr[..., 3:]
    x = F.normalize(x, dim=-1)
    z = torch.cross(x, y)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat2quat(mat)


def inv_affine(mat):
    """
    Calculate the inverse of any affine transformation
    """
    affine = torch.zeros((mat.shape[:2] + (1, 4)))
    affine[..., 3] = 1
    vert_mat = torch.cat((mat, affine), dim=2)
    vert_mat_inv = torch.inverse(vert_mat)
    return vert_mat_inv[..., :3, :]


def inv_rigid_affine(mat):
    """
    Calculate the inverse of a rigid affine transformation
    """
    res = mat.clone()
    res[..., :3] = mat[..., :3].transpose(-2, -1)
    res[..., 3] = -torch.matmul(res[..., :3], mat[..., 3].unsqueeze(-1)).squeeze(-1)
    return res


def generate_pose(batch_size, device, uniform=False, factor=1, root_rot=False, n_bone=None, ee=None):
    if n_bone is None: n_bone = 24
    if ee is not None:
        if root_rot:
            ee.append(0)
        n_bone_ = n_bone
        n_bone = len(ee)
    axis = torch.randn((batch_size, n_bone, 3), device=device)
    axis /= axis.norm(dim=-1, keepdim=True)
    if uniform:
        angle = torch.rand((batch_size, n_bone, 1), device=device) * np.pi
    else:
        angle = torch.randn((batch_size, n_bone, 1), device=device) * np.pi / 6 * factor
        angle.clamp(-np.pi, np.pi)
    poses = axis * angle
    if ee is not None:
        res = torch.zeros((batch_size, n_bone_, 3), device=device)
        for i, id in enumerate(ee):
            res[:, id] = poses[:, i]
        poses = res
    poses = poses.reshape(batch_size, -1)
    if not root_rot:
        poses[..., :3] = 0
    return poses


def slerp(l, r, t, unit=True):
    """
    :param l: shape = (*, n)
    :param r: shape = (*, n)
    :param t: shape = (*)
    :param unit: If l and h are unit vectors
    :return:
    """
    eps = 1e-8
    if not unit:
        l_n = l / torch.norm(l, dim=-1, keepdim=True)
        r_n = r / torch.norm(r, dim=-1, keepdim=True)
    else:
        l_n = l
        r_n = r
    omega = torch.acos((l_n * r_n).sum(dim=-1).clamp(-1, 1))
    dom = torch.sin(omega)

    flag = dom < eps

    res = torch.empty_like(l_n)
    t_t = t[flag].unsqueeze(-1)
    res[flag] = (1 - t_t) * l_n[flag] + t_t * r_n[flag]

    flag = ~ flag

    t_t = t[flag]
    d_t = dom[flag]
    va = torch.sin((1 - t_t) * omega[flag]) / d_t
    vb = torch.sin(t_t * omega[flag]) / d_t
    res[flag] = (va.unsqueeze(-1) * l_n[flag] + vb.unsqueeze(-1) * r_n[flag])
    return res


def slerp_quat(l, r, t):
    """
    slerp for unit quaternions
    :param l: (*, 4) unit quaternion
    :param r: (*, 4) unit quaternion
    :param t: (*) scalar between 0 and 1
    """
    t = t.expand(l.shape[:-1])
    flag = (l * r).sum(dim=-1) >= 0
    res = torch.empty_like(l)
    res[flag] = slerp(l[flag], r[flag], t[flag])
    flag = ~ flag
    res[flag] = slerp(-l[flag], r[flag], t[flag])
    return res


# def slerp_6d(l, r, t):
#     l_q = repr6d2quat(l)
#     r_q = repr6d2quat(r)
#     res_q = slerp_quat(l_q, r_q, t)
#     return quat2repr6d(res_q)


def interpolate_6d(input, size):
    """
    :param input: (batch_size, n_channels, length)
    :param size: required output size for temporal axis
    :return:
    """
    batch = input.shape[0]
    length = input.shape[-1]
    input = input.reshape((batch, -1, 6, length))
    input = input.permute(0, 1, 3, 2)     # (batch_size, n_joint, length, 6)
    input_q = repr6d2quat(input)
    idx = torch.tensor(list(range(size)), device=input_q.device, dtype=torch.float) / size * (length - 1)
    idx_l = torch.floor(idx)
    t = idx - idx_l
    idx_l = idx_l.long()
    idx_r = idx_l + 1
    t = t.reshape((1, 1, -1))
    res_q = slerp_quat(input_q[..., idx_l, :], input_q[..., idx_r, :], t)
    res = quat2repr6d(res_q)  # shape = (batch_size, n_joint, t, 6)
    res = res.permute(0, 1, 3, 2)
    res = res.reshape((batch, -1, size))
    return res
