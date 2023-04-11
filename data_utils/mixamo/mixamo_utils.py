import numpy as np

# rotations(24Xfeats)+contact(4Xfeats)+pos(3)+padding(3 for 6d, 1 for quat)
MIXAMO_TRAJ_MASK = np.asarray([True] * (2) + [True] * (4) + [False] * (174 - 6 - 6) + [True] + [False] + [True] + [False] * (3))

# use this to take upper from reference
def mixamo_upper_body_mask(repr='repr6d'):
    feats = 6 if repr =='repr6d' else 4
    return np.asarray([False] * (feats) + [False] * (10 * feats) + [True] * (13 * feats) + [False] * (4 * feats + 3 + (feats-3)))

# use this to take lower from reference
def mixamo_lower_body_mask(repr='repr6d'):
    feats = 6 if repr =='repr6d' else 4
    return np.asarray([True] * (feats) + [True] * (10 * feats) + [False] * (13 * feats) + [True] * (4 * feats + 3 + (feats-3)))


