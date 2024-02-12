"""
Imported & adapted from https://github.com/mailys-hau/echovox
"""
import numpy as np
import trimesh as tm



def get_voxidx(mesh, inc, vinfo):
    is_inside = lambda i: np.all((0 <= i) & (i < vinfo.shape), axis=1)
    # Subdivide mesh so you have at least one point per voxel
    verts, _ = tm.remesh.subdivide_to_size(mesh.vertices + inc, mesh.faces,
                                           max_edge=vinfo.spacing / 2,
                                           max_iter=20)
    # Translate to DICOMs coordinate. Directions given by line => right multiplication
    verts = (verts - vinfo.origin) @ np.linalg.inv(vinfo.directions)
    # Convert to voxel indexes (that represent mitral valve)
    idx = np.round(verts * vinfo.shape).astype(int) # Scale to number of voxels
    # Ensure all indexes are inside input voxel grid
    return np.unique(idx[is_inside(idx)], axis=0)
