module BiomassFromLiDAR

using MultiScaleTreeGraph
using Statistics: mean
using CSV
using DataFrames

export compute_all_mtg_data
export bind_csv_files
export segmentize_mtgs
export compute_volume
export NRMSE, RMSE, EF, nRMSE
export structural_model!

function structural_model!(mtg, fresh_density, dry_density, first_cross_section=nothing; model=structural_model_faidherbia)
    if (:radius in names(mtg))
        transform!(
            mtg,
            :radius => (x -> x * 2 * 1000) => :diameter_ps3d,
            :radius => (x -> π * (x * 1000)^2) => :cross_section_ps3d,
            symbol="N"
        ) # diameter in mm
    end

    # Step 1: computes the length of each node:
    transform!(mtg, compute_length_coord => :length, symbol="N") # length is in meters

    transform!(
        mtg,
        (node -> sum(filter(x -> x !== nothing, descendants!(node, :length, symbol="N", self=true)))) => :pathlength_subtree,
        symbol="N",
        filter_fun=x -> x[:length] !== nothing
    )

    # Identify which node is a segment root:
    transform!(mtg, is_seg => :is_segment, symbol="N")
    transform!(mtg, segment_index_on_axis => :segment_index_on_axis, symbol="N")

    transform!(
        mtg,
        (node -> length(descendants!(node, :length, symbol="N", self=true, filter_fun=is_seg))) => :segment_subtree,
        nleaves! => :number_leaves,
        symbol="N"
    )

    branching_order!(mtg, ascend=false)
    # We use basipetal topological order (from tip to base) to allow comparisons between branches of
    # different ages (the last emitted segment will always be of order 1).

    # Use the first cross-section for the first value to apply the pipe-model:
    if first_cross_section === nothing
        first_cross_section = π * ((descendants(mtg, :diameter_ps3d, ignore_nothing=true, recursivity_level=5)[1] / 2.0)^2)
    end

    transform!(mtg, (node -> pipe_model!(node, first_cross_section)) => :cross_section_pipe)

    # Adding the cross_section to the root:
    append!(
        mtg,
        (
            cross_section=first_cross_section,
            cross_section_pipe=first_cross_section,
            cross_section_sm=first_cross_section
        )
    )

    # Compute the cross-section using the structural model:
    transform!(mtg, model => :cross_section_sm, symbol="N")

    # Compute the diameters:
    transform!(mtg, :cross_section_pipe => (x -> sqrt(x / π) * 2.0) => :diameter_pipe, symbol="N")
    transform!(mtg, :cross_section_sm => (x -> sqrt(x / π) * 2.0) => :diameter_sm, symbol="N")

    # Compute the radius
    transform!(mtg, :diameter_pipe => (x -> x / 2) => :radius_pipe, symbol="N")
    transform!(mtg, :diameter_sm => (x -> x / 2) => :radius_sm, symbol="N")

    # Recompute the volume:
    compute_volume_stats(x, var) = x[var] * x[:length]

    transform!(mtg, (node -> compute_volume_stats(node, :cross_section_sm)) => :volume_sm, symbol="N") # volume in mm3
    transform!(mtg, (node -> compute_volume_stats(node, :cross_section_pipe)) => :volume_pipe_mod, symbol="N") # volume in mm3
    transform!(mtg, (node -> compute_volume_stats(node, :cross_section_ps3d)) => :volume_ps3d, symbol="N") # volume in mm3

    # And the biomass:
    transform!(mtg, (node -> node[:volume_sm] * fresh_density * 1e-3) => :fresh_mass, symbol="N") # in g
    mtg.fresh_mass = sum(descendants(mtg, :fresh_mass, symbol="N"))
    transform!(mtg, (node -> node[:volume_sm] * dry_density * 1e-3) => :dry_mass, symbol="N") # in g
    mtg.dry_mass = sum(descendants(mtg, :dry_mass, symbol="N"))

    transform!(mtg, (node -> node[:volume_pipe_mod] * fresh_density * 1e-3) => :fresh_mass_pipe_mod, symbol="N") # in g
    mtg.fresh_mass_pipe_mod = sum(descendants(mtg, :fresh_mass_pipe_mod, symbol="N"))
    transform!(mtg, (node -> node[:volume_pipe_mod] * dry_density * 1e-3) => :dry_mass_pipe_mod, symbol="N") # in g
    mtg.dry_mass_pipe_mod = sum(descendants(mtg, :dry_mass_pipe_mod, symbol="N"))

    transform!(mtg, (node -> node[:volume_ps3d] * fresh_density * 1e-3) => :fresh_mass_ps3d, symbol="N") # in g
    mtg.fresh_mass_ps3d = sum(descendants(mtg, :fresh_mass_ps3d, symbol="N"))
    transform!(mtg, (node -> node[:volume_ps3d] * dry_density * 1e-3) => :dry_mass_ps3d, symbol="N") # in g
    mtg.dry_mass_ps3d = sum(descendants(mtg, :dry_mass_ps3d, symbol="N"))

    # Clean-up the cached variables:
    clean_cache!(mtg)
end

function structural_model_faidherbia(node)
    max(
        0.0,
        0.817204 * node[:cross_section_pipe] +
        13.4731 * node[:number_leaves] -
        6.71983 * node[:segment_subtree]
    )
end

function compute_cross_section_all(x, var=:cross_section)
    if symbol(x) == "A"
        desc_cross_section = descendants(x, var, symbol="S", recursivity_level=1)
        if length(desc_cross_section) > 0
            return desc_cross_section[1]
        else
            @warn "$(x.name) has no descendants with a value for $var."
        end
    else
        x[var]
    end
end

function compute_volume_subtree(x)
    volume_descendants = filter(x -> x !== nothing, descendants!(x, :volume, symbol="S", self=true))
    length(volume_descendants) > 0 ? sum(volume_descendants) : nothing
end

function compute_cross_section(x)
    if x[:diameter] !== nothing
        π * ((x[:diameter] / 2.0)^2)
    end
end

function compute_cross_section_children(x)
    cross_section_child = filter(x -> x !== nothing, descendants!(x, :cross_section, symbol="S", recursivity_level=1))

    return length(cross_section_child) > 0 ? sum(cross_section_child) : nothing
end

function compute_cross_section_leaves(x)
    cross_section_leaves = filter(x -> x !== nothing, descendants!(x, :cross_section; filter_fun=isleaf))

    return length(cross_section_leaves) > 0 ? sum(cross_section_leaves) : nothing
end

function compute_volume(x)
    if x[:diameter] !== nothing && x[:length] !== nothing
        π * ((x[:diameter] / 2.0)^2) * x[:length]
    end
end

function compute_diameter(x)
    if x[:diameter] === nothing
        diams = [x[:diameter_50_1], x[:diameter_50_2], x[:diameter_70_1], x[:diameter_70_2]]
        filter!(x -> x !== nothing, diams)
        if length(diams) > 0
            return mean(diams)
        else
            return nothing
        end
    else
        return x[:diameter]
    end
end

function get_axis_length(x)
    axis_length = ancestors(x, :axis_length, symbol="A", recursivity_level=1)
    if length(axis_length) > 0
        axis_length[1]
    else
        nothing
    end
end

function compute_length(x)
    if x[:length] === nothing
        x[:length_mm]
    else
        return x[:length] * 10.0
    end
end

function compute_axis_length(x)
    length_descendants = filter(x -> x !== nothing, descendants!(x, :length, symbol="S", link=("/", "<"), all=false))
    length(length_descendants) > 0 ? sum(length_descendants) : nothing
end

function compute_dry_w(x)
    if x[:dry_weight_p1] !== nothing
        x[:dry_weight_p1]
    elseif x[:dry_weight_p2] !== nothing
        x[:dry_weight_p2]
    end
end

function compute_density(x)
    if x[:fresh_density] !== nothing
        x[:fresh_density]
    elseif x[:dry_weight] !== nothing
        x[:dry_weight] / x[:volume_bh]
    end
end

function compute_subtree_length!(x)
    length_descendants = filter(x -> x !== nothing, descendants!(x, :length, symbol="S", self=true))
    length(length_descendants) > 0 ? sum(length_descendants) : nothing
end

function compute_data_mtg!(mtg)
    transform!(
        mtg,
        (node -> sum(descendants!(node, :length, symbol="S", self=true, ignore_nothing=true))) => :pathlength_subtree,
        symbol="S",
        filter_fun=(node -> node[:length] !== nothing)
    )

    # Number of segments each segment bears
    transform!(
        mtg,
        (node -> length(descendants!(node, :length, symbol="S", self=true))) => :segment_subtree,
        nleaves! => :number_leaves,
        symbol="S",
    )

    branching_order!(mtg, ascend=false)
    # We use basipetal topological order (from tip to base) to allow comparisons between branches of
    # different ages (the last emitted segment will always be of order 1).

    # Compute the index of each segment on the axis in a basipetal way (from tip to base)
    transform!(
        mtg,
        (node -> length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false))) => :n_segments,
        symbol="A"
    )

    # now use n_segments to compute the index of the segment on the axis (tip = 1, base = n_segments)
    transform!(
        mtg,
        (node -> ancestors(node, :n_segments, symbol="A")[1]) => :n_segments_axis,
        (node -> isleaf(node) ? 1 : length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false)) + 1) => :segment_index_on_axis,
        symbol="S"
    )

    # Compute the total length of the axis in mm:
    transform!(
        mtg,
        compute_axis_length => :axis_length,
        symbol="A"
    )

    # Associate the axis length to each segment:
    transform!(mtg, get_axis_length => :axis_length, symbol="S")
    transform!(mtg, compute_volume => :volume, symbol="S") # volume of the segment in mm3

    transform!(mtg, compute_cross_section => :cross_section, symbol="S") # area of segment cross section in mm2
    transform!(mtg, compute_cross_section_children => :cross_section_children, symbol="S") # area of segment cross section in mm2

    # Cross section of the terminal nodes for each node
    transform!(mtg, compute_cross_section_leaves => :cross_section_leaves, symbol="S")


    # Volume of wood the section bears (all the sub-tree):
    transform!(mtg, compute_volume_subtree => :volume_subtree, symbol="S")

    # How many leaves the sibling of the node has:
    transform!(mtg, (node -> sum(nleaves_siblings!(node))) => :nleaves_siblings)

    # How many leaves the node has in proportion to its siblings + itself:
    transform!(mtg, (node -> node[:number_leaves] / (node[:nleaves_siblings] + node[:number_leaves])) => :nleaf_proportion_siblings, symbol="S")

    first_cross_section = filter(x -> x !== nothing, descendants(mtg, :cross_section, recursivity_level=5))[1]
    transform!(mtg, (node -> pipe_model!(node, first_cross_section)) => :cross_section_pipe)

    # Adding the cross_section to the root:
    append!(mtg, (cross_section=first_cross_section,))
    # Compute the cross-section for the axes nodes using the one measured on the S just below:
    transform!(mtg, compute_cross_section_all => :cross_section_all)

    # Use the pipe model, but only on nodes with a cross_section <= 314 (≈20mm diameter)
    transform!(mtg, (node -> pipe_model!(node, :cross_section_all, 314, allow_missing=true)) => :cross_section_pipe_50)

    mtg[:length] = sum(descendants(mtg, :length, symbol="S"))

    # Clean-up the cached variables:
    clean_cache!(mtg)

    return mtg
end

function compute_all_mtg_data(mtg_file, new_mtg_file, csv_file)
    # Import the mtg file:
    mtg = read_mtg(mtg_file)

    transform!(
        mtg,
        :mass_dry_sample_g => :dry_weight,
        :mass_g => :fresh_mass, # total mass of the branch
    )

    transform!(mtg, [:dry_weight, :volume_sample_cm3] => ((x, y) -> x / y) => :density, symbol="S", ignore_nothing=true)
    transform!(mtg, [:mass_fresh_sample_g, :volume_sample_cm3] => ((x, y) -> x / y) => :density_fresh, symbol="S", ignore_nothing=true)
    transform!(mtg, :circonference_mm => (x -> x / π) => :diameter, ignore_nothing=true) # diameter of the segment in mm

    if length(mtg) > 3
        # These are the complete mtgs, with the full topology and dimensions.
        # The others are only partial mtgs with the total biomass.
        transform!(mtg, :length_mm => :length)
        transform!(mtg, :diameter_mm => :diameter, symbol="S", ignore_nothing=true) # diameter of the segment in mm

        # Compute extra data:
        compute_data_mtg!(mtg)
    else
        #! The diameter was not measured on these branches, we use the one from the point cloud
        # transform!(mtg, :circonference_mm => (x -> (x / 10) / π) => :diameter, ignore_nothing=true) # diameter of the segment in mm
        if mtg.circonference_mm !== nothing # Only for faidherbia
            mtg.diameter = (mtg.circonference_mm / 10) / π
            mtg.cross_section = compute_cross_section(mtg)
        end
    end

    # write the resulting mtg to disk:
    write_mtg(new_mtg_file, mtg)

    # And the resulting DataFrame to a csv file:
    df =
        DataFrame(
            mtg,
            [
                :density, :density_fresh, :length, :diameter, :axis_length, :branching_order,
                :segment_index_on_axis, :mass_g, :volume, :volume_subtree, :cross_section,
                :cross_section_children, :cross_section_leaves, :n_segments_axis,
                :number_leaves, :pathlength_subtree, :segment_subtree,
                :cross_section_pipe, :cross_section_pipe_50, :nleaf_proportion_siblings,
                :nleaves_siblings, :cross_section_all, :comment, :id_cor
            ])

    CSV.write(csv_file, df[:, Not(:tree)])
end




###############################################
# Functions used in 2-model_diameter.jl
###############################################

function bind_csv_files(csv_files)
    dfs = []
    for i in csv_files
        df_i = CSV.read(i, DataFrame)
        df_i[:, :branch] .= splitext(basename(i))[1]

        transform!(
            df_i,
            :branch => ByRow(x -> x[5:end-1]) => :tree
        )

        rename!(
            df_i,
            :branch => :unique_branch
        )

        transform!(
            df_i,
            :unique_branch => ByRow(x -> x[end]) => :branch
        )
        push!(dfs, df_i)
    end

    df = dfs[1]
    for i in 2:length(dfs)
        df = vcat(df, dfs[i])
    end

    return df
end



###############################################
# Functions used in 3-mtg_plantscan3d_to_segments
###############################################

function segmentize_mtgs(in_folder, out_folder)
    # Listing the mtg files in the folder:
    mtg_files = filter(x -> splitext(basename(x))[2] in [".mtg"], readdir(in_folder))

    # Modifying the format of the MTG to match the one from the field, i.e. with segments and axis instead of nodes
    for i in mtg_files
        enrich_plantscan3d_mtg(
            joinpath(in_folder, i),
            joinpath(out_folder, i),
        )
    end
end

"""
    segmentize_mtg(in_file, out_file)

Transform the input mtg from plantscan3d into an mtg with segments and axis. Segments are
nodes describing the portion of the branch between two branching points. Axis is the
upper-scale grouping following segments, *i.e.* segments with a "/" or "<" link.
"""
function enrich_plantscan3d_mtg(in_file, out_file)
    # in_folder = joinpath("0-data", "3-mtg_lidar_plantscan3d", "1-raw_output")
    # out_folder = joinpath("0-data", "3-mtg_lidar_plantscan3d", "3-raw_output_segmentized")
    # out_file = joinpath(out_folder, mtg_files[1])
    # in_file = joinpath(in_folder, mtg_files[1])
    mtg = read_mtg(in_file)

    # Compute internode length and then cumulate the lenghts when deleting.

    # Step 1: computes the length of each node:
    transform!(mtg, compute_length_coord => :length, scale=2) # length is in meters

    # Step 2: cumulate the length of all nodes in a segment for each segment node:
    transform!(mtg, cumul_length_segment => :length, scale=2, filter_fun=is_seg)
    # And add a lenght of 0 for the first segment:
    mtg[1][:length] = 0.0

    # Step 3: delete nodes to make the mtg as the field measurements: with nodes only at in_filtering points
    mtg = delete_nodes!(mtg, filter_fun=is_segment!, scale=(1, 2))

    # Insert a new scale: the Axis.
    # First step we put all nodes at scale 3 instead of 2:
    transform!(mtg, (node -> scale!(node, 3)), scale=2)
    mtg[:scales] = scales(mtg)
    # 2nd step, we add axis nodes (scale 2) branching each time there's a branching node:
    template = MutableNodeMTG("+", "A", 0, 2)
    insert_parents!(mtg, template, scale=3, link="+")
    # And before the first node decomposing the plant:
    insert_parents!(mtg, NodeMTG("/", "A", 1, 2), scale=3, link="/", all=false)

    # 3d step, we change the branching nodes links to decomposition:
    transform!(mtg, (node -> link!(node, "/")), scale=3, link="+")

    # Fourth step, we rename the nodes symbol into segments "S":
    transform!(mtg, (node -> symbol!(node, "S")), symbol="N")

    # And the plant symbol as the plant name:
    symbol_from_file = splitext(replace(basename(out_file), "_" => ""))[1]

    # If the file name ends with a number we need to add something to not mistake it with an index
    if match(r"[0-9]+$", symbol_from_file) !== nothing
        symbol_from_file *= "whole"
    end

    symbol!(mtg, symbol_from_file)

    # Updating the symbols in the root node:
    mtg[:symbols] = symbols(mtg)

    # Last step, we add the index as in the field, *i.e.* the axis nodes are indexed following
    # their topological order, and the segments are indexed following their position on the axis:
    transform!(mtg, (x -> 1) => :index, symbol="A")
    transform!(mtg, A_indexing => :index, symbol="A")
    transform!(mtg, (node -> index!(node, node[:index])), symbol="A")

    # NB: this is done in 3 steps because the index is not a node attribute at first (it
    # is in the MTG field). So we first initialize the index to 1, then we update it with the
    # indexing function (A_indexing), then we update the index from the MTG field with the
    # index attribute.

    # Remove the index from the nodes attributes (only temporary):
    traverse!(mtg, node -> pop!(node, :index))

    # Set back the root node with no indexing:
    index!(mtg, -9999)
    transform!(mtg, (node -> index!(node, S_indexing(node))), scale=3)

    # Delete the old length of the nodes (length_node) from the attributes:
    transform!(mtg, (x -> x[:length_node] === nothing ? nothing : pop!(getfield(x, :attributes), :length_node)))

    # Write MTG back to file:
    write_mtg(out_file, mtg)
end


"""
    compute_length_coord(node)

Compute node length as the distance between itself and its parent.
"""
function compute_length_coord(node; x=:XX, y=:YY, z=:ZZ)
    parent_node = parent(node)
    if !isroot(parent_node)
        sqrt(
            (parent_node[x] - node[x])^2 +
            (parent_node[y] - node[y])^2 +
            (parent_node[z] - node[z])^2
        )
    else
        0.0
    end
end

"""
    is_seg(x)

Tests if a node is a segment node. A segment node is a node:
    - at a branching position, *i.e.*, a parent of more than one children node
    - a leaf.
"""
is_seg(x) = isleaf(x) || (!isroot(x) && (length(children(x)) > 1 || link(x[1]) == "+"))

"""
    segment_index_on_axis(node, symbol="N")

Compute the index of a segment node on the axis. The computation is basipetal, starting from tip to base. 
The index is the position of the segment on the axis.
"""
function segment_index_on_axis(node, symbol="N")
    isleaf(node) ? 1 : sum(descendants!(node, :is_segment, symbol=symbol, link=("/", "<"), all=false, type=Bool)) + 1
end

"""
    cumul_length_segment(node)

Cumulates the lengths of segments inside a segment. Only does it if the node is considered
as a segment, else returns 0.
"""
function cumul_length_segment(node, length_name=:length_node)
    if is_seg(node)

        length_ancestors =
            [
                node[length_name],
                ancestors(
                    node,
                    length_name,
                    filter_fun=x -> !is_seg(x),
                    scale=2,
                    all=false)...
            ]
        # NB: we don't use self = true because it would trigger a stop due to all = false
        filter!(x -> x !== nothing, length_ancestors)

        sum(length_ancestors) * 1000.0
    else
        0.0
    end
end

function A_indexing(node)
    parent_index = ancestors(
        node, :index,
        symbol="A",
        recursivity_level=1,
        type=Union{Int64,Nothing}
    )

    if length(parent_index) == 0
        return 1
    elseif link(node) == "+"
        return parent_index[1] + 1
    else
        return parent_index[1]
    end
end


function S_indexing(node)
    if isroot(node)
        return 0
    else
        link(node) == "/" ? 1 : index(parent(node)) + 1
    end
end

# function cross_section_stat_mod_all(node; symbol="N")
#     max(
#         0.0,
#         0.520508 * node[:cross_section_pipe] + 0.0153365 * node[:pathlength_subtree] +
#         6.38394 * node[:branching_order] + 10.9389 * node[:segment_index_on_axis] - 10.137 * node[:number_leaves] +
#         4.46843 * node[:segment_subtree]
#     )
# end
end


function cylinder_from_radius(node::MultiScaleTreeGraph.Node, xyz_attr=[:XX, :YY, :ZZ]; radius=:radius, symbol="S")
    node_start = ancestors(node, xyz_attr, recursivity_level=2, symbol=symbol)
    if length(node_start) != 0
        Cylinder(
            Point3((node_start[1][1], node_start[1][2], node_start[1][3] .+ 0.01)),
            Point3((node[xyz_attr[1]], node[xyz_attr[2]], node[xyz_attr[3]] .+ 0.01)),
            node[radius] # radius in meter
        )
    end
end


function circle_from_radius(node::MultiScaleTreeGraph.Node, xyz_attr=[:XX, :YY, :ZZ]; radius=:radius, symbol="S", radius_factor=1.0)
    node_start = ancestors(node, xyz_attr, recursivity_level=2, symbol=symbol)
    if length(node_start) != 0
        template_cyl = cylinder_from_radius(node, xyz_attr, radius=radius, symbol=symbol)
        top = extremity(template_cyl)
        new_end_point = top + direction(template_cyl) * radius_factor # Scale the direction vector to the desired length

        # Create a new cylinder with the same end point, but adding a new end point that gives the width of the circle:
        new_cyl = Cylinder(top, new_end_point, Makie.radius(template_cyl))
    end
end

function draw_skeleton!(axis, node::MultiScaleTreeGraph.Node, xyz_attr=[:XX, :YY, :ZZ]; symbol="S", color=:slategrey, linewidth)
    node_start = ancestors(node, xyz_attr, recursivity_level=2, symbol=symbol)
    if length(node_start) != 0
        lines!(
            axis,
            [node_start[1][1], node[:XX]],
            [node_start[1][2], node[:YY]],
            [node_start[1][3], node[:ZZ]] .+ 0.01,
            color=color,
            linewidth=linewidth
        )
    end
end