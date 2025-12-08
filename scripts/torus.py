#!/usr/bin/env python3

import numpy as np
from scipy.spatial import Delaunay
import trimesh
import meshio

#nu=50
#nv=50
nu=15
nv=15
R=3.0
r=1.0
bump_amplitude=0.3
bump_count=5
"""
Generate a wiggly torus mesh with vertex normals.

Parameters:
- nu, nv: number of points in u and v directions (total = nu * nv)
- R: base major radius
- r: minor radius (tube radius)
- bump_amplitude: amplitude of the bumps on the major circle
- bump_count: number of bumps around the torus

Returns:
- vertices: (N, 3) array of 3D vertex coordinates
- faces: (M, 3) array of triangle indices
- normals: (N, 3) array of vertex normals
"""

u = np.linspace(0, 2 * np.pi, nu, endpoint=True)
v = np.linspace(0, 2 * np.pi, nv, endpoint=True)
uu, vv = np.meshgrid(u, v, indexing='ij')

# Bumpy major radius
R_bumpy = R + bump_amplitude * np.sin(bump_count * uu)

# Parametric torus with bumps
x = (R_bumpy + r * np.cos(vv)) * np.cos(uu)
y = (R_bumpy + r * np.cos(vv)) * np.sin(uu)
z = r * np.sin(vv)

# Flatten to list of 3D points
vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

# Triangulate the parametric grid (2D)
points_2d = np.stack([uu.ravel(), vv.ravel()], axis=-1)
tri = Delaunay(points_2d)
faces = tri.simplices
print(faces.shape)
#revert normals
faces[:, 1], faces[:, 2] = faces[:, 2], faces[:, 1].copy()

meshio.Mesh(vertices, {"triangle": faces}).write("torus_255.ply", binary=False)


# Use trimesh to compute normals
#mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
#mesh.export("out.ply")
#normals = mesh.vertex_normals

#return vertices.T, faces.T, normals.T
