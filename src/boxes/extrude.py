"""
Imported & adapted from https://github.com/mailys-hau/echovox
"""
import numpy as np
import trimesh as tm

from boxes.utils import get_voxidx
from utils.tee import VoxelTEE



def _extrude(mesh, voxinfo, increments):
    # Voxelize every increment around surface with proper spacing to get a full volume
    # Accumulate pointsin subdivided meshes, it'll be your positive voxels
    allidx = []
    evec = mesh.vertex_normals # Extrude vector
    for i in range(increments.shape[-1]):
        inc = increments[:,i]
        allidx.append(get_voxidx(mesh, inc * evec, voxinfo))
    allidx = np.unique(np.concatenate(allidx, axis=0), axis=0)
    voxgrid = np.zeros(voxinfo.shape, dtype=bool)
    # Set voxels inside mitral annulus to be True
    voxgrid[allidx[:, 0], allidx[:, 1], allidx[:, 2]] = 1
    return VoxelTEE(voxgrid, voxinfo=voxinfo)


def rectangle_extrude(mesh, voxinfo, thickness=3, midaxis=1):
    mbox = mesh.bounding_box
    bounds = np.stack([mbox.vertices[0], mbox.vertices[-1]])
    bounds[:,midaxis] = bounds[:,midaxis].mean() # Middle along Y-axis or Z-axis
    middle = tm.creation.box(bounds=bounds) # Middle mesh
    increments = np.stack([np.arange(-thickness / 2, thickness / 2 + vr, vr)
                            for vr in voxinfo.spacing])
    return _extrude(middle, voxinfo, increments)

def normal_extrude(mesh, voxinfo, thickness=3, midaxis=-1):
    """ Taken from `echovox` """
    # Voxelize every increment around surface with proper spacing to get a full volume
    increments = np.stack([np.arange(-thickness / 2, thickness / 2 + vr, vr)
                            for vr in voxinfo.spacing])
    return _extrude(mesh, voxinfo, increments)
