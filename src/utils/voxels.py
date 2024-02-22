import numpy as np
import scipy.ndimage as scpn
import warnings

from utils.misc import get_affine



warnings.simplefilter("default", UserWarning)




def resample(vgrid, ores, nres):
    oshape = np.array(vgrid.shape)
    nshape = np.round(oshape * ores / nres)
    zoom = nshape / oshape
    vgrid = scpn.zoom(vgrid, zoom)
    return vgrid


def mid_vox_grid(vgrid, deep_axis=2):
    """ Find the middle surface of a voxel grid along `deep_axis` """
    out = np.zeros_like(vgrid)
    x, y, z = list(vgrid.shape)
    if deep_axis == 1: # GE uses Y-axis as deep_axis
        pass #TODO
    max_dist = 0
    for i in range(x):
        for j in range(y):
            if vgrid[i, j].sum() == 0:
                continue # Shortcut
            first, last = -1, -1
            for k in range(z):
                if vgrid[i, j, k]:
                    if first < 0:
                        first = k # Top outer voxel
                    last = k
            if 0 < first and 0 < last: # Security
                if last - first > max_dist:
                    max_dist = last - first
                mid = int((last - first) / 2)
                out[i, j, mid] = True
    # We lost the connectivity, but since we're making bounding boxes it's fine
    return out, max_dist


class VoxelInfo:
    def __init__(self, origin=None, directions=None, affine=None, spacing=[0.5, 0.5, 0.5],
                 unit="mm", shape=(), colormap=None, frame_time=-1):
        if origin is None and directions is None and affine is None:
            raise ValueError("Either define affine or (origin & directions). Got neither.")
        if unit != "mm" and unit != 'm':
            raise ValueError(f"Voxel grids cancan only be in mm or m. Got {self.unit}.")
        self.origin, self.directions = origin, directions
        self.affine = affine
        self.spacing = spacing if isinstance(spacing, np.ndarray) \
                                else np.array(spacing)
        self.unit, self.shape = unit, shape
        self.colormap, self.frame_time = colormap, frame_time
        if self.affine is None:
            self._affine()
        else: # Implies that directions and origin are lacking
            self.directions = self.affine[:3,:3]
            self.origin = self.affine[:3, -1]


    def to_mm(self):
        if self.unit == 'm':
            self.unit = "mm"
            self.directions *= 1000
            self.origin *= 1000
            self.spacing *= 1000
            self._affine()

    def to_m(self):
        if self.unit == "mm":
            self.unit = 'm'
            self.directions /= 1000
            self.origin /= 1000
            self.spacing /= 1000
            self._affine()

    @property
    def unit_as_nifti(self):
        return 1 if self.unit == 'm' else 2 # Necessarily m or mm

    def _affine(self):
        # Needed to save in NIFTI format
        # Affine should be used to convert to RAS system
        #dirs = self.directions / np.linalg.norm(self.directions, axis=0)
        self.affine = get_affine(self.origin, self.directions, self.spacing)

    def add_shape(self, shape):
        if self.shape != ():
            warnings.warn("Shape was already define, overwritting it.", UserWarning)
        self.shape = shape
