using DataFrames
using Statistics
using MultiScaleTreeGraph

csv_files =
    filter(
        x -> endswith(x, ".csv"), # all MTGs
        # x -> endswith(x, r"tree[1,3].\.csv"), # train only on 2020 MTGs
        readdir(joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched"), join=true)
    )

df = let
    x = dropmissing(bind_csv_files(csv_files), :cross_section)
    filter!(x -> ismissing(x.comment) || !(x.comment in ["casse", "CASSE", "broken", "AVORTE", "Portait aussi un axe peut-être cassé lors de la manip"]), x)
    filter!(y -> y.symbol == "S", x)
    # Remove this segment because we probably measured it wrong (or on a protrusion), it has a circonference of 220 mm while its predecessor has 167 mm and successor 163 mm.
    filter!(row -> !(row.tree == "1_156" && row.id == 6), x)
    x
end

df_density = let
    df_density_fresh = combine(
        groupby(dropmissing(df, :density_fresh), :unique_branch),
        :density_fresh => mean => :fresh_density,
        :density_fresh => std => :fresh_density_sd,
    )
    df_density_dry = combine(
        groupby(dropmissing(df, :density), :unique_branch),
        :density => mean => :dry_density,
        :density => std => :dry_density_sd,
    )

    leftjoin(df_density_fresh, df_density_dry, on=:unique_branch)
end

dir_path_lidar = joinpath("0-data", "3-mtg_lidar_plantscan3d", "4-corrected_segmentized")
dir_path_lidar_raw = nothing
dir_path_manual = joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched")
dir_path_lidar_new = nothing
mtg_files =
    filter(
        x -> splitext(basename(x))[2] in [".mtg"],
        readdir(dir_path_lidar)
    )

branches = first.(splitext.(mtg_files))

branch = branches[17]

function compute_volume_model(branch, dir_path_lidar, dir_path_manual, df_density)
    # Compute the average density:
    if branch in df_density.unique_branch
        dry_density = filter(x -> x.unique_branch == branch, df_density).dry_density[1]
        fresh_density = filter(x -> x.unique_branch == branch, df_density).fresh_density[1]
    else
        dry_density = mean(df_density.dry_density)
        fresh_density = mean(df_density.fresh_density)
    end

    # Importing the mtg from the manual measurement data:
    mtg_manual = read_mtg(joinpath(dir_path_manual, replace(branch, r"_cor$" => "") * ".mtg"))
    # Re-estimate the volume of the branch from the volume of its segments:
    mtg_manual[:volume] = sum(descendants!(mtg_manual, :volume, symbol="S"))

    mtg_lidar_model = read_mtg(joinpath(dir_path_lidar, branch * ".mtg"))

    compute_data_mtg_lidar!(mtg_lidar_model, fresh_density, dry_density, nothing)

    return (mtg_manual, mtg_lidar_model)
end

mtg_manual = read_mtg(joinpath(dir_path_manual, branch * ".mtg"))
descendants(mtg_manual, :diameter)

mtg_lidar_model = read_mtg(joinpath(dir_path_lidar, branch * ".mtg"))
findall(x -> x === NaN, descendants(mtg_lidar_model, :radius, symbol="S"))

if branch in df_density.unique_branch
    dry_density = filter(x -> x.unique_branch == branch, df_density).dry_density[1]
    fresh_density = filter(x -> x.unique_branch == branch, df_density).fresh_density[1]
else
    dry_density = mean(df_density.dry_density)
    fresh_density = mean(df_density.fresh_density)
end

function cross_section_stat_mod(node, model)

    # Get the node attributes as a DataFrame for the model:
    attr_names = coefnames(model)
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

    predict(model, DataFrame(attr_values...))[1]
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
