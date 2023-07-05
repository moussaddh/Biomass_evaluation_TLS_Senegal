module BiomassFromLiDAR

using MultiScaleTreeGraph
using Statistics: mean
using CSV
using DataFrames
import MLBase

export compute_all_mtg_data
export bind_csv_files
export segmentize_mtgs
export compute_volume, compute_var_axis
export RMSE, EF, nRMSE
export compute_volume_model, volume_stats

###############################################
# Functions used in 1-compute_field_mtg_data.jl
###############################################

function compute_data_mtg(mtg)
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
        symbol="A",
    )

    # now use n_segments to compute the index of the segment on the axis (tip = 1, base = n_segments)
    @mutate_mtg!(
        mtg,
        n_segments_axis = ancestors(node, :n_segments, symbol="A")[1],
        segment_index_on_axis = length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false)) + 1,
        symbol = "S"
    )

    # Compute the total length of the axis in mm:
    @mutate_mtg!(
        mtg,
        axis_length = compute_axis_length(node),
        symbol = "A"
    )

    # Associate the axis length to each segment:
    @mutate_mtg!(mtg, axis_length = get_axis_length(node), symbol = "S")

    @mutate_mtg!(mtg, volume = compute_volume(node), symbol = "S") # volume of the segment in mm3

    # @mutate_mtg!(mtg, volume = compute_var_axis(node), symbol = "A") # volume of the axis in mm3

    @mutate_mtg!(mtg, cross_section = compute_cross_section(node), symbol = "S") # area of segment cross section in mm2
    @mutate_mtg!(mtg, cross_section_children = compute_cross_section_children(node), symbol = "S") # area of segment cross section in mm2

    # Cross section of the terminal nodes for each node
    @mutate_mtg!(mtg, cross_section_leaves = compute_cross_section_leaves(node), symbol = "S")

    # Volume of wood the section bears (all the sub-tree):
    @mutate_mtg!(mtg, volume_subtree = compute_volume_subtree(node), symbol = "S")

    # How many leaves the sibling of the node has:
    @mutate_mtg!(mtg, nleaves_siblings = sum(nleaves_siblings!(node)))

    # How many leaves the node has in proportion to its siblings + itself:
    @mutate_mtg!(mtg, nleaf_proportion_siblings = node[:number_leaves] / (node[:nleaves_siblings] + node[:number_leaves]), symbol = "S")

    first_cross_section = filter(x -> x !== nothing, descendants(mtg, :cross_section, recursivity_level=5))[1]
    @mutate_mtg!(mtg, cross_section_pipe = pipe_model!(node, first_cross_section))

    # Adding the cross_section to the root:
    append!(mtg, (cross_section=first_cross_section,))
    # Compute the cross-section for the axes nodes using the one measured on the S just below:
    @mutate_mtg!(mtg, cross_section_all = compute_cross_section_all(node))

    mtg[:length] = sum(descendants(mtg, :length, symbol="S"))

    # Clean-up the cached variables:
    clean_cache!(mtg)

    return mtg
end

function compute_cross_section_all(x, var=:cross_section)
    if x.MTG.symbol == "A"
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

"""
    compute_var_axis(x, vol_col = :volume)

Sum a variable over an axis alone, excluding the axis it bears itself.
"""
function compute_var_axis(x, vol_col=:volume)
    sum(descendants!(x, vol_col, symbol="S", link=("/", "<"), all=false))
end

"""
    compute_A1_axis_from_start(x, vol_col = :volume; id_cor_start)

Compute the sum of a variable over the axis starting from node that has `id_cor_start` value.
"""
function compute_A1_axis_from_start(x, vol_col=:volume; id_cor_start)
    length_gf_A1 = descendants!(x, vol_col, symbol="S", link=("/", "<"), all=false)
    id_cor_A1 = descendants!(x, :id_cor, symbol="S", link=("/", "<"), all=false)
    sum(length_gf_A1[findfirst(x -> x == id_cor_start, id_cor_A1):end])
end



function compute_var_axis_A2(x, vol_col=:volume)
    sum(descendants!(x, vol_col, symbol="S"))
end



function get_axis_length(x)
    axis_length = ancestors(x, :axis_length, symbol="A", recursivity_level=1)
    if length(axis_length) > 0
        axis_length[1]
    else
        nothing
    end
end



function compute_axis_length(x)
    length_descendants = filter(x -> x !== nothing, descendants!(x, :length, symbol="S", link=("/", "<"), all=false))
    length(length_descendants) > 0 ? sum(length_descendants) : nothing
end




function compute_subtree_length!(x)
    length_descendants = filter(x -> x !== nothing, descendants!(x, :length, symbol="S", self=true))
    length(length_descendants) > 0 ? sum(length_descendants) : nothing
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
        transform!(mtg, :diameter_mm => :diameter, symbol="S") # diameter of the segment in mm

        # Compute extra data:
        compute_data_mtg(mtg)
    end
    # write the resulting mtg to disk:
    write_mtg(new_mtg_file, mtg)

    # And the resulting DataFrame to a csv file:
    df =
        DataFrame(
            mtg,
            [
                :density, :density_fresh, :length, :diameter, :axis_length, :branching_order,
                :segment_index_on_axis, :fresh_mass, :volume, :volume_subtree, :cross_section,
                :cross_section_children, :cross_section_leaves, :n_segments_axis,
                :number_leaves, :pathlength_subtree, :segment_subtree,
                :cross_section_pipe, :cross_section_pipe_50, :nleaf_proportion_siblings,
                :nleaves_siblings, :cross_section_all, :comment, :id_cor
            ])

    # Remove columns that contain only missing values:
    select!(df, Not(names(df, Missing)))

    CSV.write(csv_file, df[:, Not(:tree)])
end


function compute_all_mtg_data_ps3d(mtg_file, new_mtg_file, csv_file)
    # Import the mtg file:
    mtg = read_mtg(mtg_file)

    transform!(mtg, :radius => (x -> x === NaN ? 0.0 : x * 2.0 * 1000.0) => :diameter, symbol="S") # diameter of the segment in mm

    # Compute extra data:
    compute_data_mtg(mtg)
    sum(descendants(mtg, :radius, symbol="S"))

    # write the resulting mtg to disk:
    write_mtg(new_mtg_file, mtg)

    # And the resulting DataFrame to a csv file:
    df =
        DataFrame(
            mtg,
            [
                :length, :diameter, :axis_length, :branching_order,
                :segment_index_on_axis, :volume, :volume_subtree, :cross_section,
                :cross_section_children, :cross_section_leaves, :n_segments_axis,
                :number_leaves, :pathlength_subtree, :segment_subtree,
                :cross_section_pipe, :nleaf_proportion_siblings,
                :nleaves_siblings, :cross_section_all
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
        segmentize_mtg(
            joinpath(in_folder, i),
            joinpath(out_folder, i),
        )

        compute_all_mtg_data_ps3d(
            joinpath(out_folder, i),
            joinpath(out_folder, i),
            joinpath(out_folder, splitext(basename(i))[1] * ".csv")
        )
    end
end

"""
    segmentize_mtg(in_file, out_file)

Transform the input mtg from plantscan3d into an mtg with segments and axis. Segments are
nodes describing the portion of the branch between two branching points. Axis is the
upper-scale grouping following segments, *i.e.* segments with a "/" or "<" link.
"""
function segmentize_mtg(in_file, out_file)
    mtg = read_mtg(in_file)
    # Compute internode length and then cumulate the lenghts when deleting.

    # Step 1: computes the length of each node:
    @mutate_mtg!(mtg, length_node = compute_length_coord(node), scale = 2) # length is in meters

    # Step 3: cumulate the length of all nodes in a segment for each segment node:
    @mutate_mtg!(mtg, length = cumul_length_segment(node), scale = 2, filter_fun = is_seg)
    # And add a lenght of 0 for the first segment:
    mtg[1][:length] = 0.0

    # Trying to remove the NaNs in the radius column by taking the first non-NaN value in the children that follow (on the same axis):
    traverse!(mtg, scale=2) do node
        if !isleaf(node) && !isroot(node) && length(node.children) == 1
            # We keep the root and the leaves, but want to delete the nodes with no branching.
            # We recognise them because they have only one child. Also we want to keep the very
            # first node even if it has only one child.


            if node[:radius] === NaN
                # First, we try to take the radius of the first child that has a non-NaN radius:
                child_follow_radius = descendants(node, :radius, link="<", filter_fun=node -> node[:radius] !== NaN)
                new_radius = findfirst(x -> x !== NaN, child_follow_radius)

                if new_radius === nothing
                    # If all children have NaN radius, we take try with the parent, but it has to be a segment node:
                    parent_follow_radius = ancestors(node, :radius, link="<", filter_fun=node -> node[:radius] !== NaN && length(node.children) == 1, all=false)
                    new_radius = findfirst(x -> x !== NaN, parent_follow_radius)

                    # If we can't find a non-NaN radius, we just skip this node and keep the NaN radius:
                    if new_radius === nothing
                        return
                    end
                end

                node[:radius] = new_radius
            end
        end
    end

    # Step 4: delete nodes to make the mtg as the field measurements: with nodes only at in_filtering points
    mtg = delete_nodes!(mtg, filter_fun=is_segment!, scale=(1, 2))

    # Insert a new scale: the Axis.
    # First step we put all nodes at scale 3 instead of 2:
    @mutate_mtg!(mtg, node.MTG.scale = 3, scale = 2)

    # 2nd step, we add axis nodes (scale 2) branching each time there's a branching node:
    template = MutableNodeMTG("+", "A", 0, 2)
    insert_parents!(mtg, template, scale=3, link="+")
    # And before the first node decomposing the plant:
    insert_parents!(mtg, NodeMTG("/", "A", 1, 2), scale=3, link="/", all=false)

    # 3d step, we change the branching nodes links to decomposition:
    @mutate_mtg!(mtg, node.MTG.link = "/", scale = 3, link = "+")

    # Fourth step, we rename the nodes symbol into segments "S":
    @mutate_mtg!(mtg, node.MTG.symbol = "S", symbol = "N")

    # And the plant symbol as the plant name:
    symbol_from_file = splitext(replace(basename(out_file), "_" => ""))[1]

    # If the file name ends with a number we need to add something to not mistake it with an index
    if match(r"[0-9]+$", symbol_from_file) !== nothing
        symbol_from_file *= "whole"
    end

    mtg.MTG.symbol = symbol_from_file

    # Last step, we add the index as in the field, *i.e.* the axis nodes are indexed following
    # their topological order, and the segments are indexed following their position on the axis:
    @mutate_mtg!(mtg, node.MTG.index = A_indexing(node))

    # Set back the root node with no indexing:
    mtg.MTG.index = nothing

    @mutate_mtg!(mtg, node.MTG.index = S_indexing(node), scale = 3)

    # Delete the old length of the nodes (length_node) from the attributes:
    traverse!(mtg, x -> x[:length_node] === nothing ? nothing : pop!(x.attributes, :length_node))

    # Write MTG back to file:
    write_mtg(out_file, mtg)
end


"""
    compute_length_coord(node)

Compute node length as the distance between itself and its parent.
"""
function compute_length_coord(node)
    if !isroot(node.parent)
        sqrt(
            (node.parent[:XX] - node[:XX])^2 +
            (node.parent[:YY] - node[:YY])^2 +
            (node.parent[:ZZ] - node[:ZZ])^2
        )
    else
        0.0
    end
end

"""
    is_seg(x)

Is a node also a segment node ? A segment node is a node at a branching position, or at the
first (or last) position in the tree.
"""
is_seg(x) = isleaf(x) || (!isroot(x) && (length(x.children) > 1 || x[1].MTG.link == "+"))

"""
    cumul_length_segment(node)

Cumulates the lengths of segments inside a segment. Only does it if the node is considered
as a segment, else returns 0.
"""
function cumul_length_segment(node)
    if is_seg(node)

        length_ancestors =
            [
                node[:length_node],
                ancestors(
                    node,
                    :length_node,
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
    if isroot(node)
        return 1
    else
        node.MTG.link == "+" ? node.parent.MTG.index + 1 : node.parent.MTG.index
    end
end


function S_indexing(node)
    if isroot(node)
        return 0
    else
        node.MTG.link == "/" ? 1 : node.parent.MTG.index + 1
    end
end

"""
    RMSE(obs,sim)

Returns the Root Mean Squared Error between observations `obs` and simulations `sim`.
The closer to 0 the better.
"""
function RMSE(obs, sim, digits=2)
    return round(sqrt(sum((obs .- sim) .^ 2) / length(obs)), digits=digits)
end

"""
    nRMSE(obs,sim)

Returns the normalized Root Mean Squared Error between observations `obs` and simulations `sim`.
The closer to 0 the better.
"""
function nRMSE(obs, sim; digits=2)
    return round(sqrt(sum((obs .- sim) .^ 2) / length(obs)) / (findmax(obs)[1] - findmin(obs)[1]), digits=digits)
end

"""
    EF(obs,sim)

Returns the Efficiency Factor between observations `obs` and simulations `sim` using NSE (Nash-Sutcliffe efficiency) model.
More information can be found at https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient.
The closer to 1 the better.
"""
function EF(obs, sim, digits=2)
    SSres = sum((obs - sim) .^ 2)
    SStot = sum((obs .- mean(obs)) .^ 2)
    return round(1 - SSres / SStot, digits=digits)
end


function compute_data_mtg_lidar!(mtg, fresh_density, dry_density, model)

    # All variables related to radius come from ps3d:
    transform!(
        mtg,
        :diameter => :diameter_ps3d,
        :cross_section => :cross_section_ps3d,
        :volume => :volume_ps3d,
        symbol="S"
    )

    transform!(mtg,
        [:cross_section_pipe, :length] => ((x, y) -> x * y) => :volume_pipe,
        symbol="S"
    )

    transform!(mtg, (x -> cross_section_stat_mod(x, model)) => :cross_section_stat_mod, symbol="S")

    transform!(mtg,
        [:cross_section_stat_mod, :length] => ((x, y) -> x * y) => :volume_stat_mod,
        symbol="S"
    )

    mtg[:volume_ps3d] = sum(descendants(mtg, :volume_ps3d, symbol="S"))
    mtg[:volume_stat_mod] = sum(descendants(mtg, :volume_stat_mod, symbol="S"))
    mtg[:volume_pipe] = sum(descendants(mtg, :volume_pipe, symbol="S"))

    mtg[:fresh_mass_ps3d] = mtg[:volume_ps3d] * fresh_density * 1e-3 # in g
    mtg[:fresh_mass_stat_mod] = mtg[:volume_stat_mod] * fresh_density * 1e-3 # in g
    mtg[:fresh_mass_pipe] = mtg[:volume_pipe] * fresh_density * 1e-3 # in g

    mtg[:dry_mass_ps3d] = mtg[:volume_ps3d] * dry_density * 1e-3 # in g
    mtg[:dry_mass_stat_mod] = mtg[:volume_stat_mod] * dry_density * 1e-3 # in g
    mtg[:dry_mass_pipe] = mtg[:volume_pipe] * dry_density * 1e-3 # in g

    # Clean-up the cached variables:
    clean_cache!(mtg)

    return nothing
end

function cross_section_stat_mod(node, model)

    # Get the node attributes as a DataFrame for the model:
    attr_names = MLBase.coefnames(model)
    attr_values = []

    for i in attr_names
        if i == "(Intercept)"
            continue
        end
        node_val = node[i]
        if node_val === nothing
            # No missing values allowed for predicting
            return missing
        end

        push!(attr_values, i => node_val)
    end

    MLBase.predict(model, DataFrame(attr_values...))[1]
end

function compute_volume_model(branch, dir_path_lidar, dir_path_manual, df_density, model)
    # Compute the average density:
    dry_density = filter(x -> x.unique_branch == replace(branch, r"_cor$" => ""), df_density).dry_density[1]
    fresh_density = filter(x -> x.unique_branch == replace(branch, r"_cor$" => ""), df_density).fresh_density[1]

    # Importing the mtg from the manual measurement data:
    mtg_manual = read_mtg(joinpath(dir_path_manual, replace(branch, r"_cor$" => "") * ".mtg"))
    # Re-estimate the volume of the branch from the volume of its segments:
    mtg_manual[:volume] = sum(descendants!(mtg_manual, :volume, symbol="S"))

    mtg_lidar_model = read_mtg(joinpath(dir_path_lidar, branch * ".mtg"))

    compute_data_mtg_lidar!(mtg_lidar_model, fresh_density, dry_density, model)

    return (mtg_manual, mtg_lidar_model)
end

filter_A1_A2(x) = x.MTG.symbol == "A" && (x.MTG.index == 1 || x.MTG.index == 2)
filter_A1_A2_S(x) = x.MTG.symbol == "S" || filter_A1_A2(x)


function volume_stats(mtg_manual, mtg_lidar_ps3d_raw, mtg_lidar_model, df_density)
    df_lidar_raw = DataFrame(mtg_lidar_ps3d_raw, [:volume_ps3d, :volume_stat_mod, :volume_pipe_mod, :volume_pipe_mod_50, :length, :cross_section_stat_mod])
    df_lidar_model = DataFrame(mtg_lidar_model, [:volume_ps3d, :volume_stat_mod, :volume_pipe_mod, :volume_pipe_mod_50, :length, :cross_section_stat_mod])
    df_manual = DataFrame(mtg_manual, [:volume_gf, :length_gap_filled, :cross_section_gap_filled])

    # Getting the densities:
    dry_density = filter(x -> x.branches == mtg_lidar_model.MTG.symbol, df_density).dry_density[1]
    fresh_density = filter(x -> x.branches == mtg_lidar_model.MTG.symbol, df_density).fresh_density[1]

    tot_lenght_lidar = sum(filter(x -> x.symbol == "S", df_lidar_model).length) / 1000 # length in m
    tot_lenght_lidar_raw = sum(filter(x -> x.symbol == "S", df_lidar_raw).length) / 1000 # length in m
    tot_lenght_manual = sum(filter(x -> x.symbol == "S", df_manual).length_gap_filled) / 1000

    tot_vol_lidar = filter(x -> x.scale == 1, df_lidar_model).volume_ps3d[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_raw = filter(x -> x.scale == 1, df_lidar_raw).volume_ps3d[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_stat_mod = filter(x -> x.scale == 1, df_lidar_model).volume_stat_mod[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_pipe_mod = filter(x -> x.scale == 1, df_lidar_model).volume_pipe_mod[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_stat_mod_raw = filter(x -> x.scale == 1, df_lidar_raw).volume_stat_mod[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_pipe_mod_raw = filter(x -> x.scale == 1, df_lidar_raw).volume_pipe_mod[1] * 1e-9 # Total volume in m3
    tot_vol_manual = filter(x -> x.scale == 1, df_manual).volume_gf[1] * 1e-9 # Total volume in m3

    # Biomass:

    # The fresh density is either taken as the average measured density at the lab or the one
    # computed from the dimension measurements and the whole branch biomass:
    actual_fresh_density = mtg_manual.attributes[:mass_g] / (tot_vol_manual * 1e6)

    dry_biomass_lidar = tot_vol_lidar * dry_density * 1000 # mass in kg
    fresh_biomass_lidar = tot_vol_lidar * fresh_density * 1000 # fresh biomass in kg
    fresh_biomass_actual_lidar = tot_vol_lidar * actual_fresh_density * 1000 # fresh biomass in kg

    dry_biomass_lidar_raw = tot_vol_lidar_raw * dry_density * 1000 # mass in kg
    fresh_biomass_lidar_raw = tot_vol_lidar_raw * fresh_density * 1000 # fresh biomass in kg
    fresh_biomass_actual_lidar_raw = tot_vol_lidar_raw * actual_fresh_density * 1000 # fresh biomass in kg

    dry_biomass_lidar_stat_mod = tot_vol_lidar_stat_mod * dry_density * 1000 # mass in kg
    fresh_biomass_lidar_stat_mod = tot_vol_lidar_stat_mod * fresh_density * 1000 # fresh biomass in kg
    # Using the density re-computed using the volume manual measurement:
    fresh_biomass_actual_stat_mod = tot_vol_lidar_stat_mod * actual_fresh_density * 1000 # fresh biomass in kg
    fresh_biomass_lidar_stat_mod_raw = tot_vol_lidar_stat_mod_raw * fresh_density * 1000 # fresh biomass in kg

    fresh_biomass_lidar_pipe_mod = tot_vol_lidar_pipe_mod * fresh_density * 1000 # fresh biomass in kg
    fresh_biomass_lidar_pipe_mod_raw = tot_vol_lidar_pipe_mod_raw * fresh_density * 1000 # fresh biomass in kg

    dry_biomass_manual = tot_vol_manual * dry_density * 1000 # mass in kg
    fresh_biomass_manual = tot_vol_manual * fresh_density * 1000 # fresh biomass in kg

    true_fresh_biomass = mtg_manual.attributes[:mass_g] / 1000

    DataFrame(
        variable=["length", "length", "volume", "volume", "volume", "volume", "volume", "volume", "biomass", "biomass", "biomass", "biomass", "biomass", "biomass"],
        model=["plantscan3d cor.", "plantscan3d raw", "plantscan3d cor.", "plantscan3d raw", "stat. model cor.", "Pipe model cor.", "stat. model raw", "Pipe model raw", "plantscan3d cor.", "plantscan3d raw", "stat. model cor.", "Pipe model cor.", "stat. model raw", "Pipe model raw"],
        measurement=[tot_lenght_manual, tot_lenght_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass],
        prediction=[tot_lenght_lidar, tot_lenght_lidar_raw, tot_vol_lidar, tot_vol_lidar_raw, tot_vol_lidar_stat_mod, tot_vol_lidar_pipe_mod, tot_vol_lidar_stat_mod_raw, tot_vol_lidar_pipe_mod_raw, fresh_biomass_lidar, fresh_biomass_lidar_raw, fresh_biomass_lidar_stat_mod, fresh_biomass_lidar_pipe_mod, fresh_biomass_lidar_stat_mod_raw, fresh_biomass_lidar_pipe_mod_raw]
    )
end
end