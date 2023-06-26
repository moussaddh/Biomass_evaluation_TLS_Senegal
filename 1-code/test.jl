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
    csv_files =
        filter(
            x -> endswith(x, ".csv"), # all MTGs
            # x -> endswith(x, r"tree[1,3].\.csv"), # train only on 2020 MTGs
            readdir(joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched"), join=true)
        )

    dfs = []
    for i in csv_files
        df_i = CSV.read(i, DataFrame, select=[:id, :symbol, :fresh_mass, :diameter, :density_fresh, :density, :comment])
        df_i[:, :branch] .= splitext(basename(i))[1]

        transform!(
            df_i,
            :branch => ByRow(x -> replace(split.(x, "_")[end], "EC" => "")) => :tree
        )

        rename!(
            df_i,
            :branch => :unique_branch
        )

        transform!(
            df_i,
            :unique_branch => ByRow(x -> x[end]) => :branch
        )

        # Some branches don't have density measurements, we set it to missing:
        if !hasproperty(df_i, :density_fresh)
            df_i[!, :density_fresh] .= missing
        end

        if !hasproperty(df_i, :density)
            df_i[!, :density] .= missing
        end

        if !hasproperty(df_i, :comment)
            df_i[!, :comment] .= ""
        end

        push!(dfs, df_i)
    end

    df = dfs[1]
    for i in 2:length(dfs)
        df = vcat(df, dfs[i])
    end

    filter!(x -> ismissing(x.comment) || !(x.comment in ["casse", "CASSE", "broken", "AVORTE", "Portait aussi un axe peut-être cassé lors de la manip"]), df)
    filter!(y -> y.symbol == "S" || y.symbol == "B", df)
    # Remove this segment because we probably measured it wrong (or on a protrusion), it has a circonference of 220 mm while its predecessor has 167 mm and successor 163 mm.
    filter!(row -> !(row.tree == "1561" && row.id == 6), df)
    df.unique_branch = lowercase.(df.unique_branch)

    df
end

df_density = let
    df_density_fresh = combine(
        groupby(df, :unique_branch),
        :density_fresh => (x -> mean(skipmissing(x))) => :fresh_density,
        :density_fresh => (x -> std(skipmissing(x))) => :fresh_density_sd,
    )

    df_density_dry = combine(
        groupby(df, :unique_branch),
        :density => (x -> mean(skipmissing(x))) => :dry_density,
        :density => (x -> std(skipmissing(x))) => :dry_density_sd,
    )

    # Put missing densities to the average value:
    df_density_fresh.fresh_density[isnan.(df_density_fresh.fresh_density)] .= mean(filter(x -> !isnan(x), df_density_fresh.fresh_density))
    df_density_fresh.fresh_density_sd[isnan.(df_density_fresh.fresh_density_sd)] .= mean(filter(x -> !isnan(x), df_density_fresh.fresh_density_sd))
    df_density_dry.dry_density[isnan.(df_density_dry.dry_density)] .= mean(filter(x -> !isnan(x), df_density_dry.dry_density))
    df_density_dry.dry_density_sd[isnan.(df_density_dry.dry_density_sd)] .= mean(filter(x -> !isnan(x), df_density_dry.dry_density_sd))

    leftjoin(df_density_fresh, df_density_dry, on=:unique_branch)
end

dir_path_lidar = joinpath("0-data", "3-mtg_lidar_plantscan3d", "4-corrected_segmentized")
dir_path_manual = joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched")
mtg_files =
    filter(
        x -> splitext(basename(x))[2] in [".mtg"],
        readdir(dir_path_lidar)
    )

branches = first.(splitext.(mtg_files))

branch = branches[1]

function compute_volume_model(branch, dir_path_lidar, dir_path_manual, df_density)
    branch = lowercase(branch)
    # Compute the average density:
    if branch in lowercase.(df_density.unique_branch)
        dry_density = filter(x -> x.unique_branch == branch, df_density).dry_density[1]
        fresh_density = filter(x -> x.unique_branch == branch, df_density).fresh_density[1]
    else
        dry_density = mean(df_density.dry_density)
        fresh_density = mean(df_density.fresh_density)
    end

    # Importing the mtg from the manual measurement data:
    mtg_manual = read_mtg(joinpath(dir_path_manual, branch * ".mtg"))
    # Re-estimate the volume of the branch from the volume of its segments:
    vols = descendants!(mtg_manual, :volume)
    if !all(isnothing.(vols))
        mtg_manual[:volume] = sum(vols)
    end
    mtg_lidar_model = read_mtg(joinpath(dir_path_lidar, branch * ".mtg"))

    mtg_lidar_model[1][1][:diameter] = first(descendants(mtg_manual, :diameter, self=true))
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

    # we force the first segment to stay at measurement:
    mtg[1][1][:cross_section_stat_mod] = mtg[1][1][:cross_section_ps3d]

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








#####################
using CSV, DataFrames
csv_files =
    filter(
        x -> endswith(x, ".csv"), # all MTGs
        # x -> endswith(x, r"tree[1,3].\.csv"), # train only on 2020 MTGs
        readdir(joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched"), join=true)
    )

dfs = []
# i = csv_files[17]
for i in csv_files
    df_i = CSV.read(i, DataFrame, select=[:fresh_mass, :diameter, :density_fresh])
    df_i[:, :branch] .= splitext(basename(i))[1]

    transform!(
        df_i,
        :branch => ByRow(x -> replace(split.(x, "_")[end], "EC" => "")) => :tree
    )

    rename!(
        df_i,
        :branch => :unique_branch
    )

    transform!(
        df_i,
        :unique_branch => ByRow(x -> x[end]) => :branch
    )

    if hasproperty(df_i, :density_fresh)
        df_i[!, :density_fresh] .= missing
    end

    push!(dfs, df_i)
end

dfs

df = dfs[1]
for i in 2:length(dfs)

    df = vcat(df, dfs[i])
end

dfs[17]