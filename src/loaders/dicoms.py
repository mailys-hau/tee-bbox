"""
Arranged from `https://github.com/mailys-hau/echovox`.
"""

import comtypes
import comtypes.client as cct
import ctypes
import numpy as np
import platform
import warnings

from pathlib import WindowsPath

from utils.lookup import LUT
from utils import VoxelTEE



Image3DAPIWin32 = None
Image3DAPIx64 = WindowsPath("C:/Users/malou/Documents/dev/Image3dAPI/x64/Image3dAPI.tlb")

warnings.simplefilter("always", RuntimeWarning)




def safe2np(safearr_ptr, copy=True):
    """ Convert a SAFEARRAY buffer to its numpy equivalent """
    # Only support 1D data for now
    assert(comtypes._safearray.SafeArrayGetDim(safearr_ptr) == 1)
    # Access underlying pointer
    data_ptr = ctypes.POINTER(safearr_ptr._itemtype_)()
    comtypes._safearray.SafeArrayAccessData(safearr_ptr, ctypes.byref(data_ptr))
    # +1 to go from inclusive to exclusive bound
    upper_bound = comtypes._safearray.SafeArrayGetUBound(safearr_ptr, 1) + 1
    lower_bound = comtypes._safearray.SafeArrayGetLBound(safearr_ptr, 1)
    array_size = upper_bound - lower_bound
    # Wrap pointer in numpy array
    arr = np.ctypeslib.as_array(data_ptr, shape=(array_size,))
    return np.copy(arr) if copy else arr

def frame2arr(frame):
    arr1d = safe2np(frame.data, copy=False)
    assert(arr1d.dtype == np.uint8) # Only tested with 1 byte element
    arr3d = np.lib.stride_tricks.as_strided(arr1d, shape=frame.dims,
                                            strides=(1, frame.stride0, frame.stride1))
    return np.copy(arr3d)


def dcm2vox(fname, time_frame, voxres):
    if "32" in platform.architecture()[0]:
        Image3dAPI = cct.GetModule(str(Image3DAPIWin32))
    else:
        Image3dAPI = cct.GetModule(str(Image3DAPIx64))
    # Create loader object
    loader = cct.CreateObject("GEHC_CARD_US.Image3dFileLoader")
    loader = loader.QueryInterface(Image3dAPI.IImage3dFileLoader)
    # Load file
    err_type, err_msg = loader.LoadFile(str(fname))
    src = loader.GetImageSource()
    # Get general information
    bbox = src.GetBoundingBox()
    origin = np.array([bbox.origin_x, bbox.origin_y, bbox.origin_z])
    dir_x = np.array([bbox.dir1_x, bbox.dir1_y, bbox.dir1_z])
    dir_y = np.array([bbox.dir2_x, bbox.dir2_y, bbox.dir2_z])
    dir_z = np.array([bbox.dir3_x, bbox.dir3_y, bbox.dir3_z])
    directions = np.stack([dir_x, dir_y, dir_z])
    res = np.round(np.linalg.norm(directions, axis=1) / voxres)
    try:
        lut = np.array(src.GetColorMap(), dtype=np.uint).astype(np.uint8)
    except:
        warnings.warn("No colormap found in DICOM file, using a generic one. See `utils.py`", RuntimeWarning)
        lut = LUT
    # Get corresponding input frame
    max_res = np.ctypeslib.as_ctypes(res.astype(np.ushort))
    fidx = src.GetFrameTimes().index(time_frame)
    frame = frame2arr(src.GetFrame(fidx, bbox, max_res))
    frame = lut[frame]
    out = VoxelTEE(frame, origin=origin, directions=directions, spacing=voxres,
                   unit='m', shape=frame.shape, colormap=lut, frame_time=time_frame)
    return out
