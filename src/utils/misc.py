"""
Imported from https://github.com/mailys-hau/echovox (except `get_affine`)
"""
import numpy as np



def label2onehot(labels, skip_classes=[0]):
    classes = [ c for c in np.unique(labels) if c not in skip_classes ]
    onehot = np.zeros([len(classes), *labels.shape], dtype=bool)
    for i, c in enumerate(classes):
        idx = np.argwhere(labels == c)
        onehot[i, idx[:,0], idx[:,1], idx[:,2]] = 1
    return onehot

def onehot2label(onehot, skip_classes=[]):
    labels = np.zeros(onehot.shape[1:], dtype=np.uint8)
    classes = [ c for c in range(1, len(onehot) + 1) if c not in skip_classes ]
    for c in classes:
        idx = np.argwhere(onehot[c - 1] == 1)
        labels[idx[:,0], idx[:,1], idx[:,2]] = c
    return labels


def get_affine(origin, directions, spacing):
    # Needed to save in NIFTI format
    if origin.ndim == 1:
        origin = np.expand_dims(origin, axis=1)
    dirs = directions * spacing
    rotation = np.array([0, 0, 0, 1])
    return np.vstack([np.hstack([dirs, origin]), rotation])
