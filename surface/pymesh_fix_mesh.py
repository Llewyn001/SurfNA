#!/usr/bin/env python
"""Run SurfDock/MaSIF-style PyMesh fix_mesh in an isolated Python environment."""

from __future__ import print_function

import argparse

import numpy as np
from numpy.linalg import norm
import pymesh


def mesh_node_count(mesh):
    return int(getattr(mesh, "num_vertices", getattr(mesh, "num_nodes", len(mesh.vertices))))


def pick_largest(mesh_or_meshes):
    if isinstance(mesh_or_meshes, (list, tuple)):
        if not mesh_or_meshes:
            raise ValueError("PyMesh returned no mesh components")
        return max(mesh_or_meshes, key=mesh_node_count)
    return mesh_or_meshes


def fix_mesh(mesh, resolution):
    bbox_min, bbox_max = mesh.bbox
    _diag_len = norm(bbox_max - bbox_min)
    target_len = resolution

    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)
    mesh, _ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, _ = pymesh.split_long_edges(mesh, target_len)

    count = 0
    num_vertices = mesh.num_vertices
    while True:
        mesh, _ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, _ = pymesh.collapse_short_edges(mesh, target_len, preserve_feature=True)
        mesh, _ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break
        num_vertices = mesh.num_vertices
        count += 1
        if count > 10:
            break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh = pick_largest(pymesh.compute_outer_hull(mesh, all_layers=True))
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh, _ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)
    mesh = pick_largest(pymesh.separate_mesh(mesh))
    return mesh


def compute_shape_index(mesh):
    mesh.add_attribute("vertex_mean_curvature")
    h = np.asarray(mesh.get_attribute("vertex_mean_curvature"), dtype=float)
    mesh.add_attribute("vertex_gaussian_curvature")
    k = np.asarray(mesh.get_attribute("vertex_gaussian_curvature"), dtype=float)
    elem = np.square(h) - k
    elem[elem < 0] = 1e-8
    k1 = h + np.sqrt(elem)
    k2 = h - np.sqrt(elem)
    denom = k1 - k2
    denom[np.abs(denom) < 1e-8] = 1e-8
    raw_si = np.arctan((k1 + k2) / denom) * (2.0 / np.pi)
    bad = int(np.count_nonzero(~np.isfinite(raw_si)))
    si = np.nan_to_num(raw_si, nan=0.0, posinf=0.0, neginf=0.0)
    return si, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mesh_res", type=float, default=1.0)
    args = parser.parse_args()

    with np.load(args.input) as data:
        vertices = np.asarray(data["vertices"], dtype=float)
        faces = np.asarray(data["faces"], dtype=np.int64)
    mesh = pymesh.form_mesh(vertices, faces)
    mesh = fix_mesh(mesh, args.mesh_res)
    si, si_bad = compute_shape_index(mesh)
    if mesh.num_vertices == 0 or mesh.num_faces == 0:
        raise ValueError("fix_mesh produced an empty mesh")
    np.savez(
        args.output,
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        si=si,
        si_bad=np.asarray([si_bad], dtype=np.int64),
    )


if __name__ == "__main__":
    main()
