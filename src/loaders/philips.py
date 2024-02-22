"""
Load input & GT from various vendors. Put everything in an RAS coordinate system.
"""

import nrrd
import numpy as np
import trimesh as tm

from scipy.ndimage import affine_transform
from skimage.measure import marching_cubes

from utils import label2onehot, get_affine, MeshTEE, VoxelInfo, VoxelTEE
from utils.voxels import mid_vox_grid, resample



def load_philips(fname, dmask, voxres, merge=True):
    iimg, info = nrrd.read(fname)
    # Should only yield one file
    gtname = next(dmask.glob(f"{fname.stem}*{fname.suffix}"))
    fid = int(gtname.stem.split('_')[-1])
    vinp = VoxelTEE(iimg[fid], origin=info["space origin"],
                    directions=info["space directions"][1:], #FIXME
                    spacing=abs(info["space directions"][1:].diagonal()),
                    shape=iimg.shape[1:], frame_time=fid)
    vinp.resample(voxres)
    gtimg, gtinfo = nrrd.read(gtname) # GT affine may be different than input, so use `gtinfo`

    # Philips GT contains 0=background, 1=MA, 2=anterior, 3=posterior
    # => Only fetch 2 and 3
    gtarr = label2onehot(gtimg, [0, 1])
    # Extract useful information
    spacing = abs(gtinfo["space directions"].diagonal())
    # Because Philips, directions is a variation of identity
    directions = gtinfo["space directions"] / spacing
    aorigin, porigin = gtinfo["space origin"], gtinfo["space origin"]

    # Resample before getting the middle yield a better surface
    ravox, rpvox = resample(gtarr[0], spacing, voxres), resample(gtarr[1], spacing, voxres)
    # Get middle surface and maximum thickness per leaflet
    (mavox, adist), (mpvox, pdist) = mid_vox_grid(ravox), mid_vox_grid(rpvox)
    # Empirically found out that this gives better placement of the box
    aorigin[-1], porigin[-1] = 2 * adist, 2 * pdist
    avox = VoxelTEE(mavox, origin=aorigin, directions=directions,
                    spacing=voxres, shape=mavox.shape, frame_time=fid)
    pvox = VoxelTEE(mpvox, origin=porigin, directions=directions,
                    spacing=voxres, shape=mpvox.shape, frame_time=fid)
    amesh, pmesh = avox.to_mesh(), pvox.to_mesh()
    mesh = MeshTEE.concat([amesh, pmesh]) if merge else [amesh, pmesh]
    return mesh, vinp
