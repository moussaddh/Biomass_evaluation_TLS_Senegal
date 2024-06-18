using MultiScaleTreeGraph, CSV, DataFrames
using GLMakie
using GeometryBasics
include("./functions.jl")
using .BiomassFromLiDAR

# Compute the biomass of each tree:
wood_density = 0.61 # t/m3

mtgs = filter(x -> endswith(basename(x), ".mtg"), readdir("0-data/3-mtg_lidar_plantscan3d/0-raw_output_tree", join=true))

trees_volume = Dict{String,Float64}()
trees_biomass = Dict{String,Float64}()

for path in mtgs
    tree_name = replace(basename(path), ".mtg" => "", "Tree_" => "")
    @show tree_name
    mtg = read_mtg(path)
    if isnan(mtg[1][:radius])
        mtg[1][:radius] = mtg[1][1][:radius]
    end

    # Compute the cross-sectional area from the radius:
    transform!(mtg[1], :radius => (r -> π * r^2) => :cross_section) # in m

    # Apply the pipe model below a certain cross section area (0.05 m² radius):
    pipe_model!(mtg[1], :cross_section, π * 0.05^2) # 0.05 is the radius below wich we recompute the cross section (in m²)

    # Compute the length of each segment of the mtg:
    transform!(mtg, BiomassFromLiDAR.compute_length_coord => :length, scale=2) # length is in meters

    # Compute the volume of each segment of the mtg based on the cross section area and the length of the segment:
    transform!(mtg[1], [:cross_section, :length] => ((c, l) -> c * l) => :volume) # in m³

    push!(trees_volume, tree_name => sum(descendants(mtg[1], :volume)))

    # Compute the biomass of each segment of the mtg based on the volume and the wood density:
    transform!(mtg[1], :volume => (v -> v * wood_density) => :biomass) # biomass is in t

    # Compute the total biomass of the tree:
    push!(trees_biomass, tree_name => sum(descendants(mtg[1], :biomass)))
end

df = DataFrame((tree=collect(keys(trees_volume)), volume=collect(values(trees_volume)), biomass=collect(values(trees_biomass))))

CSV.write("2-results/1-data/1-tree_biomass_LiDAR.csv", df)

# Vizualize the tree:

LiDAR = CSV.read("0-data/0-raw/1-lidar/Tree_EC12.txt", DataFrame, select=[1, 2, 3, 6])
rename!(LiDAR, "//X" => :X)
LiDAR = LiDAR[1:100:end, :]
mtg = read_mtg("0-data/3-mtg_lidar_plantscan3d/0-raw_output_tree/Tree_EC12.mtg")
mtg[1][:radius] = mtg[1][1][:radius]

# Compute the cross-sectional area from the radius:
transform!(mtg[1], :radius => (r -> π * r^2) => :cross_section) # in m

pipe_model!(mtg[1], :cross_section, π * 0.05^2) # 0.05 is the radius below wich we recompute the cross section (in m²)

# Compute the length of each segment of the mtg:
transform!(mtg, BiomassFromLiDAR.compute_length_coord => :length, scale=2) # length is in meters

# Compute the volume of each segment of the mtg based on the cross section area and the length of the segment:
transform!(mtg[1], [:cross_section, :length] => ((c, l) -> c * l) => :volume) # in m³

total_volume = sum(descendants(mtg[1], :volume))

wood_density = 0.61 # t/m3

# Compute the biomass of each segment of the mtg based on the volume and the wood density:
transform!(mtg[1], :volume => (v -> v * wood_density) => :biomass) # biomass is in t

# Compute the total biomass of the tree:
total_biomass = sum(descendants(mtg[1], :biomass))

branching_order!(mtg)

all_points = Point3[]
traverse!(mtg[1], symbol="N") do node
    push!(all_points, Point3(node[:XX], node[:YY], node[:ZZ]))
end
cross_section = descendants(mtg[1], :cross_section, self=true)
branching_order = descendants(mtg[1], :branching_order, self=true, type=Int64)


# Visualize the volumes of the segments:
f = Figure()
ax = Axis3(f[1, 1], azimuth=45, elevation=30)
ax2 = Axis3(f[1, 2], azimuth=45, elevation=30)

scatter!(ax, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], markersize=0.5, color=:black, overdraw=true)

traverse!(mtg[1][1], symbol="N") do node
    node_parent = node.parent
    mesh!(
        ax2,
        Cylinder(
            Point3(node_parent[:XX], node_parent[:YY], node_parent[:ZZ]),
            Point3(node[:XX], node[:YY], node[:ZZ]),
            node_parent[:radius]
        ),
        # color=:black,
        # overdraw=true
    )
end
f
