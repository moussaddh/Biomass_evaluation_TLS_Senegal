using LinearAlgebra
using MiniQhull
using CSV, DataFrames
using Makie, GLMakie, GeometryBasics, Colors
using Tables
using Meshes, MeshViz

# Plotting the results from the R alphashape3d package:
vertices = CSV.read("0-data/4-sfm_processing/1-trees/fa_g1_1561_vertices.csv", DataFrame)
tetrahedra = CSV.read("0-data/4-sfm_processing/1-trees/fa_g1_1561_tetrahedra.csv", DataFrame, types=Int)

@time begin
    vertices_makie = Makie.to_vertices(Matrix(vertices)')
    faces = [GeometryBasics.TetrahedronFace(i...) for i in eachrow(tetrahedra)]
    m = GeometryBasics.Mesh(vertices_makie, faces) # create tetrahedra mesh
    GLMakie.mesh(m, transparency=true, color=:green)
end

# Test by myself:
xyz = CSV.read("0-data/2-lidar_processing/1-trees/crowns/Tree_EC15_crown.txt", DataFrame)
# xyz = CSV.read("0-data/2-lidar_processing/1-trees/trunks/Tree_EC15_trunk.txt", DataFrame)
select!(xyz, 1:3, :Reflectance)
subsampling = 100
vert_matrix = Matrix(xyz[1:subsampling:end, 1:3])'
# vertices = Makie.to_vertices(vert_matrix)
vertices = PointSet(vert_matrix)

viz(vertices, markersize=0.1)

connectivity = delaunay(vert_matrix)
connec = [Meshes.connect(tuple(unique(i)...), Meshes.Tetrahedron) for i in eachcol(connectivity)]

m = SimpleMesh([vertices...], SimpleTopology(connec))
viz(m, showfacets=true, facetcolor=:blue, color=1:nelements(m), alpha=0.5)

viz(m.vertices)

# Python code:
#     Delaunay triangulation: for a 2D point set, a tesselation of the points into triangles (i.e. “triangulation”) where the circumscribed circle (i.e. “circumcircle”) about every triangle contains no other points in the set. For 3D points, replace “triangle” with “tetrahedron” and “circumcircle” with “circumsphere”.

#     Affinely independent: a collection of points p0, ..., pk such that all vectors vi := pi-p0 are linearly independent (i.e. in 2D not collinear, in 3D not coplanar); also called “points in general position”

#     k-simplex: the convex hull of k+1 affinely-independent points; we call the points its vertices.

#     0-simplex = point (consists of 0+1 = 1 points)
#     1-simplex = line (consists of 1+1 = 2 points)
#     2-simplex = triangle (consists of 2+1 = 3 points)
#     3-simplex = tetrahedron (consists of 3+1 = 4 points)

#     face: any simplex whose vertices are a subset of the vertices of another simplex; i.e. “a part of a simplex”

#     (geometric) simplicial complex: a collection of simplices where (1) the intersection of two simplices is a simplex, and (2) every face of a simplex is in the complex; i.e. “a bunch of simplices”

#     alpha-exposed: a simplex within a point set where the circle (2D) or ball (3D) of radius alpha through its vertices doesn’t contain any other point in the point set

#     alpha shape: the boundary of all alpha-exposed simplices of a point set

# Algorithm

# Edelsbrunner’s Algorithm is as follows:

#     Given a point cloud pts:

#         Compute the Delaunay triangulation DT of the point cloud
#         Find the alpha-complex: search all simplices in the Delaunay triangulation and (a) if any ball around a simplex is empty and has
#         a radius less than alpha (called the “alpha test”), then add it to the
#         alpha complex
#         The boundary of the alpha complex is the alpha shape

# Code

# from scipy.spatial import Delaunay
# import numpy as np
# from collections import defaultdict
# from matplotlib import pyplot as plt
# import pyvista as pv

# fig = plt.figure()
# ax = plt.axes(projection="3d")

# plotter = pv.Plotter()

# def alpha_shape_3D(pos, alpha):
#     """
#     Compute the alpha shape (concave hull) of a set of 3D points.
#     Parameters:
#         pos - np.array of shape (n,3) points.
#         alpha - alpha value.
#     return
#         outer surface vertex indices, edge indices, and triangle indices
#     """

#     tetra = Delaunay(pos)
#     # Find radius of the circumsphere.
#     # By definition, radius of the sphere fitting inside the tetrahedral needs 
#     # to be smaller than alpha value
#     # http://mathworld.wolfram.com/Circumsphere.html
#     tetrapos = np.take(pos,tetra.vertices,axis=0)
#     normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
#     ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
#     a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
#     Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
#     Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
#     Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
#     c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
#     r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))

#     # Find tetrahedrals
#     tetras = tetra.vertices[r<alpha,:]
#     # triangles
#     TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
#     Triangles = tetras[:,TriComb].reshape(-1,3)
#     Triangles = np.sort(Triangles,axis=1)
#     # Remove triangles that occurs twice, because they are within shapes
#     TrianglesDict = defaultdict(int)
#     for tri in Triangles:TrianglesDict[tuple(tri)] += 1
#     Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
#     #edges
#     EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
#     Edges=Triangles[:,EdgeComb].reshape(-1,2)
#     Edges=np.sort(Edges,axis=1)
#     Edges=np.unique(Edges,axis=0)

#     Vertices = np.unique(Edges)
#     return Vertices,Edges,Triangles

# Julia code:
xyz = CSV.read("0-data/2-lidar_processing/1-trees/crowns/Tree_EC15_crown.txt", DataFrame)
# xyz = CSV.read("0-data/2-lidar_processing/1-trees/trunks/Tree_EC15_trunk.txt", DataFrame)
select!(xyz, 1:3, :Reflectance)
subsampling = 1000
vert_matrix = Matrix(xyz[1:subsampling:end, 1:3])'
pos = vert_matrix
# vertices = Makie.to_vertices(vert_matrix)


alpha = 0.8
function alpha_shape_3d(pos, alpha)
    vertices = PointSet(pos)
    connectivity = delaunay(vert_matrix)
    tetra = [Meshes.connect(tuple(unique(i)...), Meshes.Tetrahedron) for i in eachcol(connectivity)]

    m = SimpleMesh([vertices...], SimpleTopology(connec))

    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = Meshes.vertices(m)
    normsq = sum(tetrapos .^ 2, dims=2)
    ones = ones((tetrapos.shape[0], tetrapos.shape[1], 1))
    a = det(hcat(tetrapos, ones))
    Dx = det(hcat(normsq, tetrapos[:, :, [1, 2]], ones))
    Dy = -det(hcat(normsq, tetrapos[:, :, [0, 2]], ones))
    Dz = det(hcat(normsq, tetrapos[:, :, [0, 1]], ones))
    c = det(hcat(normsq, tetrapos))
    r = sqrt.(Dx .^ 2 + Dy .^ 2 + Dz .^ 2 - 4 * a * c) ./ (2 * abs(a))

    # Find tetrahedrals
    tetras = tetra.vertices[r.<alpha, :]
    # triangles
    TriComb = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    Triangles = reshape(tetras[:, TriComb], :, 3)
    Triangles = sort(Triangles, dims=2)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = Dict{Tuple{Int,Int,Int},Int}()
    for tri in eachrow(Triangles)
        TrianglesDict[Tuple(tri)] += 1
    end
    Triangles = [tri for tri in keys(TrianglesDict) if TrianglesDict[tri] == 1]
    # edges
    EdgeComb = [(0, 1), (0, 2), (1, 2)]
    Edges = reshape(Triangles[:, EdgeComb], :, 2)
    Edges = sort(Edges, dims=2)
    Edges = unique(Edges, dims=1)

    Vertices = unique(Edges)
    return Vertices, Edges, Triangles
end

# Example

# Generate 3D points:

# np.random.seed(0) 
# points = np.random.rand(30, 3)
# ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])

# Compute the alpha shape and plot it:

# verts, edges, tri = alpha_shape_3D(points, alpha=0.1)
# ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri, linewidth=0.2, antialiased=True)

# Julia example

using GeometryBasics, Makie, Delaunay
using Random
Random.seed!(0)

points = rand(30, 3)
fig = Figure()
ax = fig[1, 1] = Axis3(fig)
scatter!(ax, points[:, 1], points[:, 2], points[:, 3])

verts, edges, tri = alpha_shape_3d(points, 0.1)
triangles = GeometryBasics.TriangleFace.(tri)
mesh!(ax, triangles, color=rand(length(triangles)), colormap=(:viridis, 0.5), transparency=true)






# faces = [TriangleFace(unique(i)) for i in eachcol(connectivity)]
# m = GeometryBasics.Mesh(vertices, faces) # create tetrahedra mesh
# Triangulate it, since Makie's mesh conversion currently doesn't handle tetrahedras itself 
# tris = GeometryBasics.triangle_mesh(m)
# fig, ax, pl = Makie.mesh(tris, color=rand(length(tris.position)), colormap=(:viridis, 0.5), transparency=true)

# Define the algorithm to compute the alpha shape for 3D points:




