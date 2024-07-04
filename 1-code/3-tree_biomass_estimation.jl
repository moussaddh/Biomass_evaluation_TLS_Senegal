using MultiScaleTreeGraph, CSV, DataFrames
using GLMakie
using Meshes
includet("./functions.jl")
using .BiomassFromLiDAR

faidherbia_model = CSV.read("2-results/1-data/4-structural_model_faidherbia.csv", DataFrame)

#! Make a macro that parses the predictors and their coefficients into a proper equation (right now you have to do it manually, copy/paste into `structural_model`):
function parse_model(df)
    predictors_coeffs = [Symbol(i.Name) => i["Coef."] for i in eachrow(df)]
    string("node[:cross_section] = ", join(["$v * node[:$k]" for (k, v) in predictors_coeffs], " + "))
end
parse_model(faidherbia_model)

function structural_model(node)
    0.7995370003578607 * node[:cross_section_pipe] +
    0.009783760726385832 * node[:axis_length] +
    8.006916976207595 * node[:number_leaves] -
    4.001410607972083 * node[:segment_subtree]
end

# Compute the biomass of each tree:
wood_density = 0.61 # t/m3

mtgs = filter(x -> endswith(basename(x), ".mtg"), readdir("0-data/3-mtg_lidar_plantscan3d/1-raw_output/faidherbia/1-tree", join=true))

df_estimation = DataFrame(:tree => String[], :model => String[], :volume => Float64[], :biomass => Float64[])
for path in mtgs # path = mtgs[1]
    tree_name = replace(basename(path), ".mtg" => "", "Tree_" => "", "_subsampled_1MPts" => "")
    @show tree_name
    mtg = read_mtg(path)

    if isnan(mtg[1][:radius])
        mtg[1][:radius] = mtg[1][1][:radius]
    end

    BiomassFromLiDAR.compute_structural_predictors!(mtg[1])
    # Compute the cross-sectional area from the radius:
    transform!(mtg[1], :radius => (r -> isnan(r) ? -1.0 : π * r^2) => :cross_section_ps3d) # in m

    transform!(mtg[1], :cross_section_ps3d => (c -> c) => :cross_section_pipe) # in m

    # Apply the pipe model below a certain cross section area (0.05 m² radius):
    pipe_model!(mtg[1], :cross_section_pipe, π * 0.05^2; allow_missing=true) # 0.05 is the radius below wich we recompute the cross section (in m²)

    # Apply the structural model:
    transform!(mtg[1], structural_model => :cross_section_sm) # in m

    # Compute the volume of each segment of the mtg based on the cross section area and the length of the segment:
    transform!(
        mtg[1],
        [:cross_section_ps3d, :length] => ((c, l) -> c * l) => :volume_ps3d,
        [:cross_section_pipe, :length] => ((c, l) -> c * l) => :volume_pipe,
        [:cross_section_sm, :length] => ((c, l) -> c * l) => :volume_sm,
    ) # in m³

    # Compute the biomass of each segment of the mtg based on the volume and the wood density:
    transform!(
        mtg[1],
        :volume_ps3d => (v -> v * wood_density) => :biomass_ps3d,
        :volume_pipe => (v -> v * wood_density) => :biomass_pipe,
        :volume_sm => (v -> v * wood_density) => :biomass_sm,
    ) # biomass is in t

    append!(
        df_estimation,
        DataFrame(
            tree=tree_name,
            model=["ps3d", "pipe", "sm"],
            volume=[sum(descendants(mtg[1], :volume_ps3d)), sum(descendants(mtg[1], :volume_pipe)), sum(descendants(mtg[1], :volume_sm))],
            biomass=[sum(descendants(mtg[1], :biomass_ps3d)), sum(descendants(mtg[1], :biomass_pipe)), sum(descendants(mtg[1], :biomass_sm))]
        )
    )
end

CSV.write("2-results/1-data/5-tree_biomass_LiDAR_faidherbia.csv", df_estimation)

# Vizualize the tree:
LiDAR = CSV.read("0-data/0-raw/1-lidar/faidherbia/full/Tree_EC12_subsampled_1MPts.txt", DataFrame, header=false, select=[1, 2, 3, 6])
rename!(LiDAR, ["x", "y", "z", "intensity"])
# LiDAR = LiDAR[1:100:end, :]
mtg = read_mtg("0-data/3-mtg_lidar_plantscan3d/1-raw_output/faidherbia/1-tree/Tree_EC12_subsampled_1MPts.mtg")
if isnan(mtg[1][:radius])
    mtg[1][:radius] = mtg[1][1][:radius]
end

BiomassFromLiDAR.compute_structural_predictors!(mtg[1])
# Compute the cross-sectional area from the radius:
transform!(mtg[1], :radius => (r -> isnan(r) ? -1.0 : π * r^2) => :cross_section_ps3d) # in m

transform!(mtg[1], :cross_section_ps3d => (c -> c) => :cross_section_pipe) # in m

# Apply the pipe model below a certain cross section area (0.05 m² radius):
# pipe_model!(mtg[1], :cross_section_pipe, π * 0.05^2; allow_missing=true) # 0.05 is the radius below wich we recompute the cross section (in m²)
root_value = π * mtg[1][:radius]^2
traverse!(mtg[1], symbol="N") do node
    pipe_model!(node, root_value, name=:cross_section_pipe)
end

# Apply the structural model:
transform!(mtg[1], structural_model => :cross_section_sm) # in m
transform!(mtg[1], :cross_section_sm => (x -> x < 0.0 ? 0.0 : x) => :cross_section_sm) # in m

# Compute the length of each segment of the mtg:
transform!(mtg, BiomassFromLiDAR.compute_length_coord => :length, scale=2) # length is in meters

all_points = Meshes.Point[]
traverse!(mtg[1], symbol="N") do node
    push!(all_points, Meshes.Point(node[:XX], node[:YY], node[:ZZ]))
end
cross_section = descendants(mtg[1], :cross_section, self=true)
branching_order = descendants(mtg[1], :branching_order, self=true, type=Int64)
nleaves(mtg)
# Visualize the volumes of the segments:
f = Figure()
ax = Axis3(f[1, 1], azimuth=45, elevation=30, aspect=(1, 1, 1))
ax2 = Axis3(f[1, 2], azimuth=45, elevation=30, aspect=(1, 1, 1))

scatter!(ax, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], markersize=0.5, color=:black, overdraw=true)

traverse!(mtg[1][1], symbol="N") do node
    node_parent = parent(node)
    viz!(
        ax2,
        Meshes.Cylinder(
            Meshes.Point(node_parent[:XX], node_parent[:YY], node_parent[:ZZ]),
            Meshes.Point(node[:XX], node[:YY], node[:ZZ]),
            node_parent[:radius],
            # sqrt(node_parent[:cross_section_pipe] / π),
            # sqrt(node_parent[:cross_section_sm] / π),
        ),
        # color=:black,
        # overdraw=true
    )
end
f
