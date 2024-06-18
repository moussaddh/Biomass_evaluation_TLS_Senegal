using MiniQhull
# using QHull
using CSV, DataFrames
using GLMakie, GeometryBasics, Colors
using Tables

xyz = CSV.read("0-data/2-lidar_processing/1-trees/crowns/Tree_EC15_crown.txt", DataFrame)
# xyz = CSV.read("0-data/2-lidar_processing/1-trees/trunks/Tree_EC15_trunk.txt", DataFrame)
select!(xyz, 1:3, :Reflectance)

connectivity = delaunay(Matrix(xyz[:, 1:3])')'



convex_hull = chull(Matrix(xyz[:, 1:3]))
convex_hull.volume
reshape(Matrix(xyz[:, 1:3]))
convex_hull.area
convex_hull.points
convex_hull.simplices
# convex_hull.vertices
# convex_hull.facets

scatter(convex_hull.points[:, 1], convex_hull.points[:, 2], convex_hull.points[:, 3], color=:grey, markersize=0.5)
mesh!(convex_hull.points, convex_hull.simplices, color=(0, 0, 0, 0.1), linewidth=0.5)
