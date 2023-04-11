##############################
#
# https://github.com/PeizhuoLi/ganimator/blob/main/models/contact.py
#
##############################
import torch


def foot_contact_by_height(pos):
    eps = 0.25
    return (-eps < pos[..., 1]) * (pos[..., 1] < eps)


def velocity(pos, padding=False):
    input_ndims = pos.ndim
    if input_ndims < 4:
        pos = pos.unsqueeze(0)

    velo = pos[:, 1:, ...] - pos[:, :-1, ...]
    velo_norm = torch.norm(velo, dim=-1)
    if padding:
        pad = torch.zeros_like(velo_norm[:, :1, :])
        velo_norm = torch.cat([pad, velo_norm], dim=1)

    if input_ndims < 4:
        velo_norm = velo_norm.squeeze(0)
    return velo_norm


def foot_contact(pos, ref_height=1., threshold=0.018):
    velo_norm = velocity(pos)
    contact = velo_norm < threshold
    contact = contact.int()
    padding = torch.zeros_like(contact)
    contact = torch.cat([padding[:1, :], contact], dim=0)
    return contact


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r


def constrain_from_contact(contact, glb, fid='TBD', L=5):
    """
    :param contact: contact label
    :param glb: original global position
    :param fid: joint id to fix, corresponding to the order in contact
    :param L: frame to look forward/backward
    :return:
    """
    T = glb.shape[0]

    for i, fidx in enumerate(fid):  # fidx: index of the foot joint
        fixed = contact[:, i]  # [T]
        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].clone()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].clone()
            avg /= (t - s + 1)

            for j in range(s, t + 1):
                glb[j, fidx] = avg.clone()
            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(L):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(L):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break
            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.clone()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.clone()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.clone()
    return glb
