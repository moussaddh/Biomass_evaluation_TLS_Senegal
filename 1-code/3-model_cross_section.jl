### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 8d606a5d-4d1f-4754-98f2-80097817c479
begin
    using CSV
    using MultiScaleTreeGraph
    using DataFrames
    using GLM
    using Statistics
    using StatsBase
    using Random
    using AlgebraOfGraphics
    using CairoMakie
	using Colors
    using ColorSchemes
    using MLBase
end

# ╔═╡ 393b8020-3743-11ec-2da9-d1600147f3d1
md"""
# Modelling cross-section surface

The purpose of this notebook is to make a model that predicts the cross-section of any segment using only features we can derive from LiDAR data.

Then we use the model to predict the cross-section of all segments from the measured branches and the whole trees.
"""

# ╔═╡ 3506b454-fb9c-4632-8dfb-15804b66add2
md"""
## Pre-requisites for the Notebook
"""

# ╔═╡ 8b711c1e-7d4e-404b-b2c8-87f536728fee
md"""
Defining the colors for the plot:
"""

# ╔═╡ 6bee7b4a-c3a1-4562-a17f-71335b8d39ae
colors = ["Stat. model" => ColorSchemes.Set2_5.colors[1], "Pipe model" => ColorSchemes.Set2_5.colors[2], "Plantscan3d" => ColorSchemes.Set2_5.colors[3]]

# ╔═╡ 220dfbff-15fc-4e75-a6a2-39e60c08e8dc
md"""
Importing the data from all MTGs, either with the full topology and dimensions, or just the branch fresh mass, base diameter and density.
"""

# ╔═╡ 492fc741-7a3b-4992-a453-fcac2bbf35ad
df = let 
	csv_files =
    filter(
        x -> endswith(x, ".csv"), # all MTGs
        # x -> endswith(x, r"tree[1,3].\.csv"), # train only on 2020 MTGs
        readdir(joinpath("../0-data", "1.2-mtg_manual_measurement_corrected_enriched"), join=true)
	) 

	dfs = []
    for i in csv_files
        df_i = CSV.read(i, DataFrame, select = [:id, :symbol, :fresh_mass, :diameter, :density_fresh, :density, :comment])
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

# ╔═╡ 75e8003d-709a-4bec-8829-979230468e33
md"""
Importing the data with the full topology and dimensions into a `DataFrame`. This data is used to define the statistical model.
"""

# ╔═╡ 068bccf7-7d01-40f5-b06b-97f6f51abcdd
md"""
!!! note
	The local functions definitions can be found at the end of the notebook
"""

# ╔═╡ 0b8d39b2-9255-4bd7-a02f-2cc055bf61fd
md"""
## Model training
"""

# ╔═╡ fa9cf6f4-eb79-4c70-ba1f-4d80b3c3e62a
md"""
First, we define which variables will be used in our model. In our case we will use all data we can derive from the LiDAR.
"""

# ╔═╡ 3d0a6b24-f11b-4f4f-b59b-5c40ea9be838
#formula_all = @formula(cross_section ~ 0 + cross_section_pipe + pathlength_subtree)
formula_all = @formula(cross_section ~ 0 + cross_section_pipe +  segment_index_on_axis + axis_length)
#formula_all = @formula(cross_section ~ 0 + cross_section_pipe + pathlength_subtree + branching_order + segment_index_on_axis + axis_length + number_leaves + segment_subtree + n_segments_axis)

# ╔═╡ b8800a04-dedb-44b3-82fe-385e3db1d0d5
md"""
### Model k-fold cross-validation

The first step is to evaluate our model using a k-fold validation to get a better estimate of its statistics. It helps us evaluate the structure of the model (*e.g.* which variables to use), while getting a good idea of its prediction capability. The model is trained and evaluated k-times on sub-samples of the data. The output statistic is then an out-of-sample statistic, which is much more conservative than the in-sample one.

Here's the nRMSE resulting from the cross-validation:
"""

# ╔═╡ bde004a8-d54c-4049-98f6-87c579785641
md"""
### Model training and evaluation on the whole data

Then, we train our model on the whole data to compute the right parameters for use in our next reconstructions.
"""

# ╔═╡ a7bf20e9-211c-4161-a5d2-124866afa76e
md"""
*Table 1. Linear model summary.*
"""

# ╔═╡ f2eb6a9d-e788-46d0-9957-1bc22a98ad5d
md"""
## Model evaluation

The model is then calibrated and evaluated on the whole dataset. Fig. 1 shows the accuracy of both modelling approaches compared to observations. Both approaches are robust and show a high modelling efficiency and relatively low error, however the pipe model presents a higher positive bias (Table 2).
"""

# ╔═╡ e2f20d4c-77d9-4b95-b30f-63febb7888c3
md"""
*Figure 1. Measured (x-axis) and predicted (y-axis) cross-section at axis scale. The prediction is done either using the statistical model (Stat. mod.), or the pipe model (Pipe mod.).*
"""

# ╔═╡ 3944b38d-f45a-4ff9-8348-98a8e04d4ad1
md"""
*Table 2. Prediction accuracy of the two modelling approach, computed with normalized root mean squared error (nRMSE, %) and modelling efficiency (EF).*
"""

# ╔═╡ 9c04906b-10cd-4c53-a879-39d168e5bd1f
md"""
## Compute the volume for LiDAR-based MTGs
"""

# ╔═╡ e5c0c40a-eb0a-4726-b58e-59c64cb39eae
md"""
### Importing the data
"""

# ╔═╡ d66aebf5-3681-420c-a342-166ea05dda2e
md"""
Importing the wood density data
"""

# ╔═╡ ecbcba9b-da36-4733-bc61-334c12045b0e
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

	df_density_fresh.measured_fresh_density .= true
	df_density_dry.measured_dry_density .= true

	# Put missing densities to the average value:
	df_density_fresh.measured_fresh_density[isnan.(df_density_fresh.fresh_density)] .= false
	df_density_fresh.fresh_density[isnan.(df_density_fresh.fresh_density)] .= mean(filter(x -> !isnan(x), df_density_fresh.fresh_density))
	
	df_density_fresh.fresh_density_sd[isnan.(df_density_fresh.fresh_density_sd)] .= mean(filter(x -> !isnan(x), df_density_fresh.fresh_density_sd))
	
	df_density_dry.measured_dry_density[isnan.(df_density_dry.dry_density)] .= false
	df_density_dry.dry_density[isnan.(df_density_dry.dry_density)] .= mean(filter(x -> !isnan(x), df_density_dry.dry_density))

	df_density_dry.dry_density_sd[isnan.(df_density_dry.dry_density_sd)] .= mean(filter(x -> !isnan(x), df_density_dry.dry_density_sd))

    leftjoin(df_density_fresh, df_density_dry, on=:unique_branch)
end

# ╔═╡ c27512cc-9c75-4dcf-9e5a-79c49e4ba478
CSV.write("../2-results/1-data/0-wood_density.csv", df_density)

# ╔═╡ f26a28b2-d70e-4543-b58e-2d640c2a0c0d
md"""
Importing the MTG files
"""

# ╔═╡ 9290e9bf-4c43-47c7-96ec-8b44ad3c6b23
begin
    dir_path_lidar = joinpath("..", "0-data", "3-mtg_lidar_plantscan3d", "4-corrected_segmentized")
    dir_path_lidar_raw = nothing
    dir_path_manual = joinpath("..", "0-data", "1.2-mtg_manual_measurement_corrected_enriched")
    dir_path_lidar_new = joinpath("..", "0-data", "3-mtg_lidar_plantscan3d", "5-corrected_enriched")
    mtg_files =
        filter(
            x -> splitext(basename(x))[2] in [".mtg"],
            readdir(dir_path_lidar)
       )
end

# ╔═╡ 466aa3b3-4c78-4bb7-944d-5d55128f8cf6
md"""
### Computing the volumes and biomass of the branches using the model
"""

# ╔═╡ a3fef18c-b3c7-4a67-9876-6af3a1968afe
md"""
#### Volume
"""

# ╔═╡ 0409c90e-fc40-4f02-8805-9feb6a7f8eb9
md"""
#### Fresh mass
"""

# ╔═╡ c9090d58-4fd6-4b4c-ad14-bf2f611cccfd
md"""
Making the same plot but separating the branches that were used for model training and model validation:
"""

# ╔═╡ 7fdbd52d-969f-47e5-9628-4de6077c8ff3
md"""
And using only the validation data:
"""

# ╔═╡ 42dc6f96-c947-476c-8073-cfe98733836c
md"""
Computing the statistics about model prediction of the branches fresh mass:
"""

# ╔═╡ b62964a9-59e8-478f-b30a-2513b6291e67
md"""
More global assessment:
"""

# ╔═╡ ddb4f5a5-5e2b-43a1-8e3f-09c3dad8870f
md"""
Same plot but showing which branch density was measured or not:
"""

# ╔═╡ 30f8608f-564e-4ffc-91b2-1f104fb46c1e
md"""
## References
"""

# ╔═╡ 5dc0b0d9-dde6-478b-9bee-b9503a3a4d82
function rename_var(x)
	if x == "pipe" || x == "Pipe mod."
		"Pipe model"
	elseif x == "ps3d"
		"Plantscan3d"
	elseif x == "stat_mod" || x == "Stat. mod."
		"Stat. model"
	else
		error("Model $x not defined.")
	end
end

# ╔═╡ ffe53b41-96bd-4f44-b313-94aabdc8b1a6
"""
	    nRMSE(obs,sim)

	Returns the normalized Root Mean Squared Error between observations `obs` and simulations `sim`.
	The closer to 0 the better.
	"""
function nRMSE(obs, sim; digits=4)
    return round(sqrt(sum((obs .- sim) .^ 2) / length(obs)) / (findmax(obs)[1] - findmin(obs)[1]), digits=digits)
end

# ╔═╡ 12d7aca9-4fa8-4461-8077-c79a99864391
"""
	    EF(obs,sim)

	Returns the Efficiency Factor between observations `obs` and simulations `sim` using [NSE (Nash-Sutcliffe efficiency) model](https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient).
	The closer to 1 the better.
	"""
function EF(obs, sim, digits=4)
    SSres = sum((obs - sim) .^ 2)
    SStot = sum((obs .- mean(obs)) .^ 2)
    return round(1 - SSres / SStot, digits=digits)
end

# ╔═╡ b6fa2e19-1375-45eb-8f28-32e1d00b5243
"""
	    RME(obs,sim)

Relative mean error between observations `obs` and simulations `sim`.
The closer to 0 the better.
"""
function RME(obs, sim, digits=4)
    return round(mean((sim .- obs) ./ obs), digits=digits)
end

# ╔═╡ 21fd863d-61ed-497e-ba3c-5f327e354cee
"""
	    Bias(obs,sim)

	Returns the bias between observations `obs` and simulations `sim`.
	The closer to 0 the better.
	"""
function Bias(obs, sim, digits=4)
    return round(mean(sim .- obs), digits=digits)
end

# ╔═╡ 0195ac30-b64f-409a-91ad-e68cf37d7c3b
function bind_csv_files(csv_files)
    dfs = []
    for i in csv_files
        df_i = CSV.read(i, DataFrame)
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
        push!(dfs, df_i)
    end

    df = dfs[1]
    for i in 2:length(dfs)
        df = vcat(df, dfs[i])
    end

    return df
end

# ╔═╡ 7e58dc6e-78ec-4ff3-8e99-97756c3a8914
df_training = let
	# Define manually which branches were measured for the full topology and dimensions:
	full_mtg_files = ["fa_g1_1561.csv", "fa_g1_tower12.csv", "fa_g2_1538.csv", "fa_g2_1606.csv", "tree_EC5.csv"]
	csv_files = [joinpath("../0-data", "1.2-mtg_manual_measurement_corrected_enriched", i) for i in full_mtg_files]
	x = dropmissing(bind_csv_files(csv_files), :cross_section)
	    filter!(x -> ismissing(x.comment) || !(x.comment in ["casse", "CASSE", "broken", "AVORTE", "Portait aussi un axe peut-être cassé lors de la manip"]), x)
	    filter!(y -> y.symbol == "S", x)
	    # Remove this segment because we probably measured it wrong (or on a protrusion), it has a circonference of 220 mm while its predecessor has 167 mm and successor 163 mm.
    filter!(row -> !(row.tree == "1561" && row.id == 6), x)
    x
end

# ╔═╡ 68fdcbf2-980a-4d44-b1f9-46b8fcd5bea1
begin
    # Define the function to compute the RMSE for each cross-validation fold:
    function compute_rmse(mod, df_ref)
        x = deepcopy(df_ref)
        x[!, "Stat. mod."] = predict(mod, x)
        x = dropmissing(x, ["Stat. mod.", "cross_section"])
        nRMSE(x[:, "Stat. mod."], x[:, :cross_section])
    end

    # Sample size
    const n = size(df_training, 1)
    Random.seed!(1234)
    # cross validation
    scores = cross_validate(
        inds -> lm(formula_all, df_training[inds, :]),        # training function
        (mod, inds) -> compute_rmse(mod, df_training[inds, :]),  # evaluation function
        n,              # total number of samples
        Kfold(n, 10)    # 10-fold cross validation plan
    )
    # get the mean and std of the scores
    (m, s) = round.(mean_and_std(scores), digits=4)

    "nRMSE: $m ± sd $s"
end

# ╔═╡ aaa829ee-ec36-4116-8424-4b40c581c2fc
model = lm(formula_all, df_training)

# ╔═╡ 4dd53a56-05dd-4244-862c-24ebaef45d52
CSV.write("../2-results/1-data/4-structural_model.csv", coeftable(model))

# ╔═╡ b49c4235-a09e-4b8c-a392-d423d7ed7d4c
df_all = let x = deepcopy(df_training)
    x[:, "Stat. mod."] = predict(model, x)
    rename!(x, Dict(:cross_section_pipe => "Pipe mod."))
    stack(
        dropmissing(x, ["Pipe mod.", "Stat. mod.", "cross_section"]),
        ["Pipe mod.", "Stat. mod."],
        [:unique_branch, :id, :symbol, :scale, :index, :parent_id, :link, :cross_section],
        variable_name=:origin,
        value_name=:cross_section_pred
    )
end;

# ╔═╡ d587f110-86d5-41c0-abc7-2671d711fbdf
article_figure_CSA = begin
    plt_cs_all =
        data(df_all) *
        (
            mapping(
                :cross_section => "Measured cross-section (mm²)",
                :cross_section => "Predicted cross-section (mm²)") * visual(Lines) +
            mapping(
                :cross_section => "Measured cross-section (mm²)",
                :cross_section_pred => "Predicted cross-section (mm²)",
                color=:origin => rename_var => "Model",
                marker=:unique_branch => "Branch") *
            visual(Scatter, markersize=20, alpha=0.8)
        )
    p = draw(plt_cs_all, palettes=(; color=colors))
end

# ╔═╡ f9e07eb8-8457-48cc-a5f9-7ebb06bbfe81
save("../2-results/2-plots/cross_section_evaluation.png", article_figure_CSA; px_per_unit=3)

# ╔═╡ dc2bd8f0-c321-407f-9592-7bcdf45f9634
stats_cross_section = begin
    stats_all =
        combine(
            groupby(df_all, [:origin]),
            [:cross_section_pred, :cross_section] => nRMSE => :nRMSE,
            [:cross_section_pred, :cross_section] => EF => :EF,
            [:cross_section_pred, :cross_section] => Bias => :Bias,
            [:cross_section_pred, :cross_section] => RME => :RME
        )
    sort(stats_all, :nRMSE)
end

# ╔═╡ b14476ab-f70b-4c22-a321-b339f94ad219
CSV.write("../2-results/1-data/2-stats_cross_section.csv", stats_cross_section)

# ╔═╡ d7a3c496-0ef0-454b-9e32-e5835928f4d5
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

# ╔═╡ eb39ed1b-6dee-4738-a762-13b759f74411
"""
	compute_A1_axis_from_start(x, vol_col = :volume; id_cor_start)

Compute the sum of a variable over the axis starting from node that has `id_cor_start` value.
"""
function compute_A1_axis_from_start(x, vol_col=:volume; id_cor_start)
    length_gf_A1 = descendants!(x, vol_col, symbol="S", link=("/", "<"), all=false)
    id_cor_A1 = descendants!(x, :id_cor, symbol="S", link=("/", "<"), all=false)
    sum(length_gf_A1[findfirst(x -> x == id_cor_start, id_cor_A1):end])
end

# ╔═╡ ee46e359-36bd-49c4-853c-d3ff29888473
function compute_var_axis_A2(x, vol_col=:volume)
    sum(descendants!(x, vol_col, symbol="S"))
end

# ╔═╡ b2e75112-be43-4df9-86df-2eeeb58f47c3
filter_A1_A2(x) = x.MTG.symbol == "A" && (x.MTG.index == 1 || x.MTG.index == 2)

# ╔═╡ b01851d1-d9d9-4016-b02e-6d3bfc449b8a
filter_A1_A2_S(x) = x.MTG.symbol == "S" || filter_A1_A2(x)

# ╔═╡ 14fde936-fa95-471a-aafb-5d69871e5a87
function compute_axis_length(x)
    length_descendants = filter(x -> x !== nothing, descendants!(x, :length, symbol="S", link=("/", "<"), all=false))
    length(length_descendants) > 0 ? sum(length_descendants) : nothing
end

# ╔═╡ e3ba9fec-c8b3-46e6-8b1d-29ab19198c9c
function get_axis_length(x)
    axis_length = ancestors(x, :axis_length, symbol="A", recursivity_level=1)
    if length(axis_length) > 0
        axis_length[1]
    else
        nothing
    end
end

# ╔═╡ 9e967170-9388-43e4-8b18-baccb18f4b4e
function compute_volume(x)
    if x[:diameter] !== nothing && x[:length] !== nothing
        π * ((x[:diameter] / 2.0)^2) * x[:length]
    end
end

# ╔═╡ 979ca113-6a22-4313-a011-0aca3cefdbf7
function compute_cross_section(x)
    if x[:diameter] !== nothing
        π * ((x[:diameter] / 2.0)^2)
    end
end

# ╔═╡ 43967391-6580-4aac-9ac1-c9effbf3c948
function compute_cross_section_children(x)
    cross_section_child = filter(x -> x !== nothing, descendants!(x, :cross_section, symbol="S", recursivity_level=1))

    return length(cross_section_child) > 0 ? sum(cross_section_child) : nothing
end

# ╔═╡ de63abdd-3879-4b7c-86f7-844f6288f987
function compute_cross_section_leaves(x)
    cross_section_leaves = filter(x -> x !== nothing, descendants!(x, :cross_section; filter_fun=isleaf))

    return length(cross_section_leaves) > 0 ? sum(cross_section_leaves) : nothing
end

# ╔═╡ d17d7e96-bd15-4a79-9ccb-6182e7d7c023
function compute_volume_subtree(x)
    volume_descendants = filter(x -> x !== nothing, descendants!(x, :volume, symbol="S", self=true))
    length(volume_descendants) > 0 ? sum(volume_descendants) : nothing
end

# ╔═╡ 27a0dcef-260c-4a0c-bef3-04a7d1b79805
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

# ╔═╡ 77486fa7-318d-4397-a792-70fd8d2148e3
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
    mtg[1][1][:cross_section_stat_mod] = mtg[1][1][:cross_section_pipe]

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

# ╔═╡ 97871566-4904-4b40-a631-98f7e837a2f4
function compute_volume_model(branch, dir_path_lidar, dir_path_manual, df_density, model)
	branch = lowercase(branch)
    # Compute the average density:
	if branch in df_density.unique_branch
    dry_density = filter(x -> x.unique_branch == branch, df_density).dry_density[1]
    fresh_density = filter(x -> x.unique_branch == branch, df_density).fresh_density[1]
	else
	    dry_density = mean(df_density.dry_density)
	    fresh_density = mean(df_density.fresh_density)
	end

    # Importing the mtg from the manual measurement data:
    mtg_manual = read_mtg(joinpath(dir_path_manual, branch * ".mtg"))
    # Re-estimate the volume of the branch from the volume of its segments:
	vols = descendants!(mtg_manual, :volume, symbol="S")
    if !all(isnothing.(vols))
        mtg_manual[:volume] = sum(vols)
    end

    mtg_lidar_model = read_mtg(joinpath(dir_path_lidar, branch * ".mtg"))

	# We take the first value of the cross-section as the one from the measurement:
	mtg_lidar_model[1][1][:diameter] = first(descendants(mtg_manual, :diameter, self=true))
	#NB: this is because the cross-section of the branches are not well estimated using ps3d already (we know that already), but it will be well estimated at tree scale because the diameter of the tree is way larger.
	
    compute_data_mtg_lidar!(mtg_lidar_model, fresh_density, dry_density, model)

    return (mtg_manual, mtg_lidar_model)
end

# ╔═╡ 0a19ac96-a706-479d-91b5-4ea3e091c3e8
function summarize_data(mtg_files, dir_path_lidar, dir_path_manual, dir_path_lidar_new, df_density, model)
    branches = first.(splitext.(mtg_files))

    evaluations = DataFrame[]

    for i in branches
        println("Computing branch $i")
        (mtg_manual, mtg_lidar_model) =
            compute_volume_model(i, dir_path_lidar, dir_path_manual, df_density, model)

        # Write the computed LiDAR MTG to disk:
        write_mtg(joinpath(dir_path_lidar_new, i * ".mtg"), mtg_lidar_model)

        push!(
            evaluations,
            DataFrame(
                :branch => i,
                :volume_ps3d => mtg_lidar_model[:volume_ps3d],
                :volume_stat_mod => mtg_lidar_model[:volume_stat_mod],
                :volume_pipe => mtg_lidar_model[:volume_pipe],
                :fresh_mass_ps3d => mtg_lidar_model[:fresh_mass_ps3d],
                :fresh_mass_stat_mod => mtg_lidar_model[:fresh_mass_stat_mod],
                :fresh_mass_pipe => mtg_lidar_model[:fresh_mass_pipe],
                :dry_mass_ps3d => mtg_lidar_model[:dry_mass_ps3d],
                :dry_mass_stat_mod => mtg_lidar_model[:dry_mass_stat_mod],
                :dry_mass_pipe => mtg_lidar_model[:dry_mass_pipe],
                :length_lidar => mtg_lidar_model[:length],
                :length_manual => mtg_manual[:length],
                :volume_manual => mtg_manual[:volume],
                :fresh_mass_manual => mtg_manual[:fresh_mass],
            )
        )
    end

    return vcat(evaluations...)
end

# ╔═╡ 87140df4-3fb5-443c-a667-be1f19b016f6
df_evaluations = summarize_data(mtg_files, dir_path_lidar, dir_path_manual, dir_path_lidar_new, df_density, model);

# ╔═╡ 915ba9a6-3ee8-4605-a796-354e7c293f55
begin
	df_vol = DataFrames.stack(select(df_evaluations, :branch, Cols(x -> startswith(x, "volume"))), Not([:branch, :volume_manual]))
	df_mass = DataFrames.stack(select(df_evaluations, :branch, Cols(x -> startswith(x, "fresh"))), Not([:branch, :fresh_mass_manual]))
	df_mass.origin .= [i in ["fa_g1_1561", "fa_g1_tower12", "fa_g2_1538", "fa_g2_1606", "Tree_EC5"] ? "Training" : "Validation" for i in df_mass.branch]
end;

# ╔═╡ 36315f52-fdb4-4872-9bcb-f5f8a9e1fb60
let
	#df_ = filter(x -> x.variable != "volume_ps3d", df_vol)
	df_ = filter(x -> !isnothing(x.volume_manual), df_vol)
    plt_volume_branches =
        data(df_) *
        (
            mapping(
                :volume_manual => "Measured volume (length x cross-section, mm³)",
                :volume_manual => "Predicted volume (mm³)") * visual(Lines) +
            mapping(
                :volume_manual => "Measured volume (length x cross-section, mm³)",
                :value => "Predicted volume (mm³)",
                color=:variable => (x -> rename_var(replace(x, "volume_" => ""))) => "Model",
                marker=:branch => "Branch"
			) *
            visual(Scatter, markersize=20, alpha=0.8)
        )
    p_vol = draw(plt_volume_branches, palettes=(; color=colors))
end

# ╔═╡ fa2acb23-a9f7-4324-99e4-923b0811591f
let
    plt_mass_branches =
        data(df_mass) *
        (
            mapping(
                :fresh_mass_manual => "Measured fresh mass (g)",
                :fresh_mass_manual => "Predicted fresh mass (g)") * visual(Lines) +
            mapping(
                :fresh_mass_manual => "Measured fresh mass (g)",
                :value => "Predicted fresh mass (g)",
                color=:variable => (x -> rename_var(replace(x, "fresh_mass_" => ""))) => "Model",
                marker=:branch => "Branch"
			) *
            visual(Scatter, markersize=20, alpha=0.8)
        )
    p_mass = draw(plt_mass_branches, palettes=(; color=colors))
end

# ╔═╡ 9dd9d67b-7856-43e1-9859-76a5463428ce
article_figure = let
    plt_mass_branches =
        data(df_mass) *
        (
            mapping(
                :fresh_mass_manual => "Measured fresh mass (g)",
                :fresh_mass_manual => "Predicted fresh mass (g)") * visual(Lines) +
            mapping(
                :fresh_mass_manual => "Measured fresh mass (g)",
                :value => "Predicted fresh mass (g)",
                color=:variable => (x -> rename_var(replace(x, "fresh_mass_" => ""))) => "Model",
                marker=:origin => "Branch"
			) *
            visual(Scatter, markersize=20, alpha=0.8)
        )
    p_mass = draw(plt_mass_branches, palettes=(; color=colors))
end

# ╔═╡ 53372fb0-c6a0-440f-acdf-bad5b205db22
save("../2-results/2-plots/biomass_evaluation.png", article_figure; px_per_unit=3)

# ╔═╡ 4dcd6a1e-ebc9-43df-a4c0-6a7702d1491e
let
    plt_mass_branches =
        data(filter(row -> row.origin == "Validation", df_mass)) *
        (
            mapping(
                :fresh_mass_manual => "Measured fresh mass (g)",
                :fresh_mass_manual => "Predicted fresh mass (g)") * visual(Lines) +
            mapping(
                :fresh_mass_manual => "Measured fresh mass (g)",
                :value => "Predicted fresh mass (g)",
                color=:variable => (x -> rename_var(replace(x, "fresh_mass_" => ""))) => "Model",
                #marker=:origin => "Branch"
			) *
            visual(Scatter, markersize=20, alpha=0.8)
        )
    p_mass = draw(plt_mass_branches, palettes=(; color=colors))
end

# ╔═╡ 8239f0f5-041e-47d0-9623-570c4acf542e
stats_evaluation = let
    stats_mass =
        combine(
            groupby(df_mass, [:variable, :origin]),
            [:value, :fresh_mass_manual] => nRMSE => :nRMSE,
            [:value, :fresh_mass_manual]  => EF => :EF,
            [:value, :fresh_mass_manual] => Bias => :Bias,
            [:value, :fresh_mass_manual] => RME => :RME
        )
    sort(stats_mass, [:origin, :nRMSE])
end

# ╔═╡ 7874d655-5551-41ea-839f-cf02a89541d8
CSV.write("../2-results/1-data/3-stats_branch_evaluation.csv", stats_evaluation)

# ╔═╡ dafd1d8c-bb3e-4862-a4d9-c235193fe850
let
    stats_mass =
        combine(
            groupby(df_mass, [:variable]),
            [:value, :fresh_mass_manual] => nRMSE => :nRMSE,
            [:value, :fresh_mass_manual]  => EF => :EF,
            [:value, :fresh_mass_manual] => Bias => :Bias,
            [:value, :fresh_mass_manual] => RME => :RME
        )
    sort(stats_mass, :nRMSE)
end

# ╔═╡ 641a6930-19e5-4775-bfd6-2483fd54737a
let
	df = leftjoin(DataFrames.transform(df_mass, :branch => (x -> lowercase.(x)) => :branch), df_density, on = :branch => :unique_branch)
    plt_mass_branches =
        data(filter(row -> row.origin == "Validation", df)) *
        (
            mapping(
                :fresh_mass_manual => "Measured fresh mass (g)",
                :fresh_mass_manual => "Predicted fresh mass (g)") * visual(Lines) +
            mapping(
                :fresh_mass_manual => "Measured fresh mass (g)",
                :value => "Predicted fresh mass (g)",
                color=:variable => (x -> rename_var(replace(x, "fresh_mass_" => ""))) => "Model",
                marker=:measured_fresh_density => "Measured density",
			) *
            visual(Scatter, markersize=20, alpha=0.8)
        )
    p_mass = draw(plt_mass_branches, palettes=(; color=colors))
end

# ╔═╡ 666e9daf-e28f-4e14-b52a-bcc6b5aadb67
cross_section_stat_mod_all = cross_section_stat_mod

# ╔═╡ 665cb43b-ab86-4001-88a3-c67ed16b28e8
function compute_data_mtg_tree!(mtg_tree, fresh_density, dry_density)

    @mutate_mtg!(mtg_tree, diameter = node[:radius] * 2 * 1000, symbol = "S") # diameter in mm

    @mutate_mtg!(
        mtg_tree,
        pathlength_subtree = sum(filter(x -> x !== nothing, descendants!(node, :length, symbol="S", self=true))),
        symbol = "S",
        filter_fun = x -> x[:length] !== nothing
    )

    @mutate_mtg!(
        mtg_tree,
        segment_subtree = length(descendants!(node, :length, symbol="S", self=true)),
        number_leaves = nleaves!(node),
        symbol = "S"
    )

    branching_order!(mtg_tree, ascend=false)
    # We use basipetal topological order (from tip to base) to allow comparisons between branches of
    # different ages (the last emitted segment will always be of order 1).

    # Compute the index of each segment on the axis in a basipetal way (from tip to base)
    @mutate_mtg!(
        mtg_tree,
        n_segments = length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false)),
        symbol = "A"
    )

    # now use n_segments to compute the index of the segment on the axis (tip = 1, base = n_segments)
    @mutate_mtg!(
        mtg_tree,
        n_segments_axis = ancestors(node, :n_segments, symbol="A")[1],
        segment_index_on_axis = length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false)) + 1,
        symbol = "S"
    )

    # Compute the total length of the axis in mm:
    @mutate_mtg!(
        mtg_tree,
        axis_length = compute_axis_length(node),
        symbol = "A"
    )

    # Associate the axis length to each segment:
    @mutate_mtg!(mtg_tree, axis_length = get_axis_length(node), symbol = "S")

    # How many leaves the sibling of the node has:
    @mutate_mtg!(mtg_tree, nleaves_siblings = sum(nleaves_siblings!(node)))

    # How many leaves the node has in proportion to its siblings + itself:
    @mutate_mtg!(mtg_tree, nleaf_proportion_siblings = node[:number_leaves] / (node[:nleaves_siblings] + node[:number_leaves]), symbol = "S")

    # Use the first cross-section for the first value to apply the pipe-model:
    first_cross_section = π * ((descendants(mtg_tree, :diameter, ignore_nothing=true, recursivity_level=5)[1] / 2.0)^2)
    @mutate_mtg!(mtg_tree, cross_section_pipe = pipe_model!(node, first_cross_section))

    # Adding the cross_section to the root:
    append!(
        mtg_tree,
        (
            cross_section=first_cross_section,
            cross_section_pipe=first_cross_section,
            cross_section_stat_mod=first_cross_section
        )
    )
    # Compute the cross-section for the axes nodes using the one measured on the S just below:
    @mutate_mtg!(mtg_tree, cross_section_pipe = compute_cross_section_all(node, :cross_section_pipe))
    @mutate_mtg!(mtg_tree, cross_section_stat_mod = cross_section_stat_mod_all(node, model), symbol = "S")

    # Add the values for the axis:
    @mutate_mtg!(mtg_tree, cross_section_stat_mod = compute_cross_section_all(node, :cross_section_stat_mod))

    # Recompute the volume:
    compute_volume_stats(x, var) = x[var] * x[:length]

    @mutate_mtg!(mtg_tree, volume_stat_mod = compute_volume_stats(node, :cross_section_stat_mod), symbol = "S") # volume in mm3
    @mutate_mtg!(mtg_tree, volume_pipe_mod = compute_volume_stats(node, :cross_section_pipe), symbol = "S") # volume in mm3

    # And the biomass:	
    @mutate_mtg!(mtg_tree, fresh_mass = node[:volume_stat_mod] * fresh_density * 1e-3, symbol = "S") # in g
    @mutate_mtg!(mtg_tree, dry_mass = node[:volume_stat_mod] * dry_density * 1e-3, symbol = "S") # in g

    @mutate_mtg!(mtg_tree, fresh_mass_pipe_mod = node[:volume_pipe_mod] * fresh_density * 1e-3, symbol = "S") # in g
    @mutate_mtg!(mtg_tree, dry_mass_pipe_mod = node[:volume_pipe_mod] * dry_density * 1e-3, symbol = "S") # in g

    # Clean-up the cached variables:
    clean_cache!(mtg_tree)
end

# ╔═╡ 073e32dd-c880-479c-8933-d53c9655a04d
function volume_stats(mtg_manual, mtg_lidar_model, df_density)
    df_lidar_model = DataFrame(mtg_lidar_model, [:volume_ps3d, :volume_stat_mod, :volume_pipe, :length, :cross_section_stat_mod])
    df_manual = DataFrame(mtg_manual, [:volume, :length, :cross_section])

    # Getting the densities:
    dry_density = filter(x -> x.unique_branch == mtg_lidar_model.MTG.symbol, df_density).dry_density[1]
    fresh_density = filter(x -> x.unique_branch == mtg_lidar_model.MTG.symbol, df_density).fresh_density[1]

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MLBase = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
MultiScaleTreeGraph = "dd4a991b-8a45-4075-bede-262ee62d5583"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AlgebraOfGraphics = "~0.6.16"
CSV = "~0.10.11"
CairoMakie = "~0.10.6"
ColorSchemes = "~3.21.0"
Colors = "~0.12.10"
DataFrames = "~1.5.0"
GLM = "~1.8.3"
MLBase = "~0.9.1"
MultiScaleTreeGraph = "~0.11.2"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "16a66f9b94413c320cd2464d5df9bff0f26e0c2d"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AlgebraOfGraphics]]
deps = ["Colors", "Dates", "Dictionaries", "FileIO", "GLM", "GeoInterface", "GeometryBasics", "GridLayoutBase", "KernelDensity", "Loess", "Makie", "PlotUtils", "PooledArrays", "PrecompileTools", "RelocatableFolders", "StatsBase", "StructArrays", "Tables"]
git-tree-sha1 = "c58b2c0f1161b8a2e79dcb1a0ec4b639c2406f15"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.6.16"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools", "SHA"]
git-tree-sha1 = "bfc7d54b3c514f8015055e6ad0d5997da64d99fc"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.10.6"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "e82c3c97b5b4ec111f3c1b55228cebc7510525a2"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.25"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c72970914c8a21b36bbc244e9df0ed1834a0360b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.95"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "0fa3b52a04a4e210aeb1626def9c90df3ae65268"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.1.0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "06bf20fcecd258eccf9a6ef7b99856a4dfe7b64c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "e17cc4dc2d0b0b568e80d937de8ed8341822de67"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.2.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "38a92e40157100e796690421e34a11c107205c86"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "97829cfda0df99ddaeaafb5b370d6cab87b7013e"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.3"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "bb198ff907228523f3dee1070ceee63b9359b6ab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "659140c9375afa2f685e37c1a0b9c9a60ef56b40"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.7"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1cf1d7dcb4bc32d7b4a5add4232db3750c27ecb4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.8.0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "678d136003ed5bceaab05cf64519e3f956ffa4ba"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.9.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "0ec02c648befc2f94156eaef13b0f38106212f3f"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.17"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "42c17b18ced77ff0be65957a591d34f4ed57c631"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.31"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "106b6aa272f294ba47e96bd3acbabdc0407b5c60"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9c6b2a4c99e7e153f3cf22e10bf40a71c7a3c6a9"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.6.1"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLBase]]
deps = ["IterTools", "Random", "Reexport", "StatsBase"]
git-tree-sha1 = "a0242608e72ba745d43ec385b2e95a562633f8fc"
uuid = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
version = "0.9.1"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "Match", "MathTeXEngine", "MiniQhull", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Setfield", "Showoff", "SignedDistanceFields", "SparseArrays", "StableHashTraits", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "a6695a632992a2e19ae1a1d0c9bee0e137e2f3cb"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.19.6"

[[deps.MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "9926529455a331ed73c19ff06d16906737a876ed"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.6.3"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test", "UnicodeFun"]
git-tree-sha1 = "8f52dbaa1351ce4cb847d95568cb29e62a307d93"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.6"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MetaGraphsNext]]
deps = ["Graphs", "JLD2", "SimpleTraits"]
git-tree-sha1 = "500e526a1f76b73ab7522f9580f86abef895de68"
uuid = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
version = "0.5.0"

[[deps.MiniQhull]]
deps = ["QhullMiniWrapper_jll"]
git-tree-sha1 = "9dc837d180ee49eeb7c8b77bb1c860452634b0d1"
uuid = "978d7f02-9e05-4691-894f-ae31a51d76ca"
version = "0.4.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultiScaleTreeGraph]]
deps = ["AbstractTrees", "DataFrames", "Dates", "DelimitedFiles", "Graphs", "MetaGraphsNext", "MutableNamedTuples", "OrderedCollections", "Printf", "SHA", "XLSX"]
git-tree-sha1 = "e7e84af1fe5cd12c7ffbadd6410bae7d248e2103"
uuid = "dd4a991b-8a45-4075-bede-262ee62d5583"
version = "0.11.2"

[[deps.MutableNamedTuples]]
git-tree-sha1 = "0faaabea6ebbfde9a5a01455f851009fb2603aac"
uuid = "af6c499f-54b4-48cc-bbd2-094bba7533c7"
version = "0.1.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "5ae7ca23e13855b3aba94550f26146c01d259267"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa4b74f80b01c6bc2b89992b861b5f210e665b5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.21+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "84a314e3926ba9ec66ac097e3635e270986b0f10"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.9+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5a6ab2f64388fd1175effdf73fe5933ef1e0bac0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "213579618ec1f42dea7dd637a42785a608b1ea9c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QhullMiniWrapper_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Qhull_jll"]
git-tree-sha1 = "607cf73c03f8a9f83b36db0b86a3a9c14179621f"
uuid = "460c41e3-6112-5d7f-b78c-b6823adb3f2d"
version = "1.0.0+1"

[[deps.Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be2449911f4d6cfddacdf7efc895eceda3eee5c1"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "8.0.1003+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0e270732477b9e551d884e6b07e23bb2ec947790"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.5"

[[deps.ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "2436b15f376005e8790e318329560dcc67188e84"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.3"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableHashTraits]]
deps = ["CRC32c", "Compat", "Dates", "SHA", "Tables", "TupleTools", "UUIDs"]
git-tree-sha1 = "0b8b801b8f03a329a4e86b44c5e8a7d7f4fe10a3"
uuid = "c5dd0088-6c3f-4803-b00e-f31a60c170fa"
version = "0.3.1"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "832afbae2a45b4ae7e831f86965469a24d1d8a83"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.26"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "8cc7a5385ecaa420f0b3426f9b0135d0df0638ed"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.2"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "8621f5c499a8aa4aa970b1ae381aae0ef1576966"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.4"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XLSX]]
deps = ["Artifacts", "Dates", "EzXML", "Printf", "Tables", "ZipFile"]
git-tree-sha1 = "d6af50e2e15d32aff416b7e219885976dc3d870f"
uuid = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"
version = "0.9.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─393b8020-3743-11ec-2da9-d1600147f3d1
# ╠═8d606a5d-4d1f-4754-98f2-80097817c479
# ╟─3506b454-fb9c-4632-8dfb-15804b66add2
# ╟─8b711c1e-7d4e-404b-b2c8-87f536728fee
# ╠═6bee7b4a-c3a1-4562-a17f-71335b8d39ae
# ╟─220dfbff-15fc-4e75-a6a2-39e60c08e8dc
# ╟─492fc741-7a3b-4992-a453-fcac2bbf35ad
# ╟─75e8003d-709a-4bec-8829-979230468e33
# ╟─7e58dc6e-78ec-4ff3-8e99-97756c3a8914
# ╟─068bccf7-7d01-40f5-b06b-97f6f51abcdd
# ╟─0b8d39b2-9255-4bd7-a02f-2cc055bf61fd
# ╟─fa9cf6f4-eb79-4c70-ba1f-4d80b3c3e62a
# ╠═3d0a6b24-f11b-4f4f-b59b-5c40ea9be838
# ╟─b8800a04-dedb-44b3-82fe-385e3db1d0d5
# ╟─68fdcbf2-980a-4d44-b1f9-46b8fcd5bea1
# ╟─bde004a8-d54c-4049-98f6-87c579785641
# ╟─a7bf20e9-211c-4161-a5d2-124866afa76e
# ╠═aaa829ee-ec36-4116-8424-4b40c581c2fc
# ╠═4dd53a56-05dd-4244-862c-24ebaef45d52
# ╟─f2eb6a9d-e788-46d0-9957-1bc22a98ad5d
# ╟─b49c4235-a09e-4b8c-a392-d423d7ed7d4c
# ╠═d587f110-86d5-41c0-abc7-2671d711fbdf
# ╟─e2f20d4c-77d9-4b95-b30f-63febb7888c3
# ╠═f9e07eb8-8457-48cc-a5f9-7ebb06bbfe81
# ╟─dc2bd8f0-c321-407f-9592-7bcdf45f9634
# ╟─3944b38d-f45a-4ff9-8348-98a8e04d4ad1
# ╠═b14476ab-f70b-4c22-a321-b339f94ad219
# ╟─9c04906b-10cd-4c53-a879-39d168e5bd1f
# ╟─e5c0c40a-eb0a-4726-b58e-59c64cb39eae
# ╟─d66aebf5-3681-420c-a342-166ea05dda2e
# ╠═ecbcba9b-da36-4733-bc61-334c12045b0e
# ╠═c27512cc-9c75-4dcf-9e5a-79c49e4ba478
# ╟─f26a28b2-d70e-4543-b58e-2d640c2a0c0d
# ╠═9290e9bf-4c43-47c7-96ec-8b44ad3c6b23
# ╟─466aa3b3-4c78-4bb7-944d-5d55128f8cf6
# ╠═87140df4-3fb5-443c-a667-be1f19b016f6
# ╠═915ba9a6-3ee8-4605-a796-354e7c293f55
# ╟─a3fef18c-b3c7-4a67-9876-6af3a1968afe
# ╟─36315f52-fdb4-4872-9bcb-f5f8a9e1fb60
# ╟─0409c90e-fc40-4f02-8805-9feb6a7f8eb9
# ╟─fa2acb23-a9f7-4324-99e4-923b0811591f
# ╟─c9090d58-4fd6-4b4c-ad14-bf2f611cccfd
# ╠═9dd9d67b-7856-43e1-9859-76a5463428ce
# ╠═53372fb0-c6a0-440f-acdf-bad5b205db22
# ╟─7fdbd52d-969f-47e5-9628-4de6077c8ff3
# ╟─4dcd6a1e-ebc9-43df-a4c0-6a7702d1491e
# ╟─42dc6f96-c947-476c-8073-cfe98733836c
# ╟─8239f0f5-041e-47d0-9623-570c4acf542e
# ╠═7874d655-5551-41ea-839f-cf02a89541d8
# ╟─b62964a9-59e8-478f-b30a-2513b6291e67
# ╟─dafd1d8c-bb3e-4862-a4d9-c235193fe850
# ╟─ddb4f5a5-5e2b-43a1-8e3f-09c3dad8870f
# ╟─641a6930-19e5-4775-bfd6-2483fd54737a
# ╟─30f8608f-564e-4ffc-91b2-1f104fb46c1e
# ╟─0a19ac96-a706-479d-91b5-4ea3e091c3e8
# ╟─5dc0b0d9-dde6-478b-9bee-b9503a3a4d82
# ╟─665cb43b-ab86-4001-88a3-c67ed16b28e8
# ╟─ffe53b41-96bd-4f44-b313-94aabdc8b1a6
# ╟─12d7aca9-4fa8-4461-8077-c79a99864391
# ╟─b6fa2e19-1375-45eb-8f28-32e1d00b5243
# ╟─21fd863d-61ed-497e-ba3c-5f327e354cee
# ╟─0195ac30-b64f-409a-91ad-e68cf37d7c3b
# ╟─77486fa7-318d-4397-a792-70fd8d2148e3
# ╟─97871566-4904-4b40-a631-98f7e837a2f4
# ╟─d7a3c496-0ef0-454b-9e32-e5835928f4d5
# ╟─eb39ed1b-6dee-4738-a762-13b759f74411
# ╟─ee46e359-36bd-49c4-853c-d3ff29888473
# ╟─b2e75112-be43-4df9-86df-2eeeb58f47c3
# ╟─b01851d1-d9d9-4016-b02e-6d3bfc449b8a
# ╟─14fde936-fa95-471a-aafb-5d69871e5a87
# ╟─e3ba9fec-c8b3-46e6-8b1d-29ab19198c9c
# ╟─9e967170-9388-43e4-8b18-baccb18f4b4e
# ╟─979ca113-6a22-4313-a011-0aca3cefdbf7
# ╟─43967391-6580-4aac-9ac1-c9effbf3c948
# ╟─de63abdd-3879-4b7c-86f7-844f6288f987
# ╟─d17d7e96-bd15-4a79-9ccb-6182e7d7c023
# ╟─27a0dcef-260c-4a0c-bef3-04a7d1b79805
# ╟─666e9daf-e28f-4e14-b52a-bcc6b5aadb67
# ╟─073e32dd-c880-479c-8933-d53c9655a04d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
