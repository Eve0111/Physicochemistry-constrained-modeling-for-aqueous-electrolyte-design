import numpy as np

try:
    from skimage.measure import marching_cubes
except ImportError:
    from skimage.measure import marching_cubes_lewiner as marching_cubes

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt

import vtk
import gc
import pyvista as pv

pv.set_jupyter_backend('static')

from matal.utils import cache_return


@cache_return
def generate_fesb_trisurf(pred_func, cutoff=0.8, n_points=251, margin=0.05):
    ## Initial grid points
    p_start, p_end = -margin, 1 + margin
    grid_space = (p_end - p_start) / (n_points - 1)
    x, y, z = np.mgrid[p_start:p_end:n_points * 1j, p_start:p_end:n_points * 1j, p_start:p_end:n_points * 1j]

    ## Predict using regressor
    prob = pred_func(np.array([x.flatten(), y.flatten(), z.flatten()]).T).reshape(x.shape).astype(float)

    ## Sum of composition fraction must be smaller or equal to 1
    prob[x + y + z > 1] = 0
    prob[x < 0] = 0
    prob[y < 0] = 0
    prob[z < 0] = 0
    prob[x > 1] = 0
    prob[y > 1] = 0
    prob[z > 1] = 0

    verts, faces, *_ = marching_cubes(prob, cutoff, spacing=(grid_space, ) * 3, allow_degenerate=False)
    verts -= margin
    
    return verts, faces


def mscatter(x, y, z, ax=None, m=None, **kw):
    ax = ax or plt.gca()
    sc = ax.scatter(x, y, z, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def skimg_tri_to_poly(verts, faces_skimg):
    nf, _ = faces_skimg.shape
    faces = []
    for i in range(nf):
        faces += [3]
        faces += list(faces_skimg[i, :])
    faces = np.asarray(faces)
    poly = pv.PolyData(verts, faces)
    return poly


def poly_to_skimg_tri(poly):
    verts, faces = poly.points, poly.faces.reshape((-1,4))[:, 1:]
    return verts, faces 


def smooth_pvs_poly(poly, sm_iters=300, sm_rlx=0.1, sm2_iters=10, sm2_rlx=0.1, 
                    dp_factor=0.95, dp_angle=180, dp_iters=50, dp_error=0.0002):
    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputData(poly)
    triangleFilter.Update()

    smoothingFilter = vtk.vtkSmoothPolyDataFilter()
    smoothingFilter.SetInputConnection(triangleFilter.GetOutputPort())
    smoothingFilter.SetNumberOfIterations(sm_iters)
    smoothingFilter.SetRelaxationFactor(sm_rlx)
    smoothingFilter.Update()

    decimator = vtk.vtkDecimatePro()
    decimator.SetTargetReduction(dp_factor)
    decimator.SetBoundaryVertexDeletion(False)
    decimator.SetFeatureAngle(dp_angle)
    decimator.MaximumIterations = dp_iters
    decimator.PreserveTopologyOn()
    decimator.SetErrorIsAbsolute(True)
    decimator.SetAbsoluteError(dp_error)
    decimator.PreserveTopologyOn()

    decimator.SetInputConnection(smoothingFilter.GetOutputPort())
    decimator.Update()

    # Additional cleaning
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(decimator.GetOutputPort())
    cleaner.Update()

    # Additional smoothing (if needed)
    smoothingFilter2 = vtk.vtkSmoothPolyDataFilter()
    smoothingFilter2.SetInputConnection(cleaner.GetOutputPort())
    smoothingFilter2.SetNumberOfIterations(sm2_iters)
    smoothingFilter2.SetRelaxationFactor(sm2_rlx)
    smoothingFilter2.Update()

    # Final triangle filter
    triangleFilter2 = vtk.vtkTriangleFilter()
    triangleFilter2.SetInputConnection(smoothingFilter2.GetOutputPort())
    triangleFilter2.Update()

    # Recalculate normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(triangleFilter2.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()

    return pv.PolyData(normals.GetOutput())


def remove_legend_alpha(leg):
    for lh in leg.legendHandles: 
        if 'Path3DCollection' in str(lh.__class__):
            lh.set_alpha(1)



@cache_return
def get_mesh_pred(model, margin=0.05, n_points=551, target='pred', margin_value=0.0):
    if target == 'pred':
        func = model.predict
    elif target == 'prob':
        class_1_idx = model.classes_.tolist().index(1)
        func = lambda x: model.predict_proba(x)[:,class_1_idx]
    elif target == 'dist':
        func = model.decision_function
    else:
        raise ValueError(target)
     
    p_start, p_end = 0 - margin, 1 + margin
    grid_space = (p_end - p_start) / (n_points - 1)
    
    x, y, z = np.mgrid[p_start:p_end:n_points * 1j, p_start:p_end:n_points * 1j, p_start:p_end:n_points * 1j]
    flattened = np.vstack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), dtype=np.float32).T
    mask = (x < 0) | (y < 0) | (z < 0) | (x > 1) | (y > 1) | (z > 1)
    shape = x.shape
    del x, y, z
    gc.collect()

    pred = func(flattened).reshape(shape).astype(np.float32)
    pred[mask] = margin_value
    return pred, grid_space, flattened

