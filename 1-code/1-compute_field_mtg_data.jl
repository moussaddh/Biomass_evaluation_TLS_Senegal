# Aim: Compute new variables in the MTG and export the results in a CSV and a new mtg.
# Author: A. Bonnet & M. Millan & R. Vezy & M. Diédhiou
# Date of creation: 22/07/2021

# Imports

using Revise
includet("./functions.jl")
using .BiomassFromLiDAR
using MultiScaleTreeGraph

# Listing the mtg files in xlsx/xlsm format:
# . veut dire ici
# .. veut dire remonter d'un cran par rapport a notre dossier actel 
# donc il faut taper "../ + ctrl + espace" pour completer le chemin du dossier ou se trouvent les fichiers MTG

mtg_files = "./0-data/1.0-mtg_manual_measurement_corrected/" .* ["fa_g1_1561.xlsm", "fa_g1_tower12.xlsm", "fa_g2_1538.xlsm", "fa_g2_1606.xlsm", "tree_EC5.xlsm"]

# Listing the mtg files in xlsx/xlsm format:
# mtg_files =
#     filter(
#         x -> splitext(basename(x))[2] in [".xlsx", ".xlsm"],
#         readdir(joinpath("0-data", "1.0-mtg_manual_measurement_corrected"), join=true)
#     )

read_mtg(mtg_files[1])
read_mtg(mtg_files[2])
read_mtg(mtg_files[3])
read_mtg(mtg_files[4])
read_mtg(mtg_files[5])


# mtg_files = "./0-data/1.1-mtg_manual_measurement_corrected_id/".*["fa_g1_1561.xlsm", "fa_g1_tower12.xlsm", "fa_g2_1538.xlsm","fa_g2_1606.xlsm","fa_g2_Tower_Ec5.xlsm"]
# Computing new variables for each mtg and saving the results in "0-data/5-enriched_manual_mtg":

for i in mtg_files
    println("Computing branch $(splitext(basename(i))[1])")

    compute_all_mtg_data(
        i,
        joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched", splitext(basename(i))[1] * ".mtg"),
        joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched", splitext(basename(i))[1] * ".csv"),
    )
end

# using MultiScaleTreeGraph
# mtg_file = "0-data/1.1-mtg_manual_measurement_corrected_id/fa_g1_tower12.xlsm"
# mtg = read_mtg(mtg_file)
# node = get_node(mtg, "node_13")
# compute_data_mtg(mtg)
# node.MTG
# node.attributes[:axis_length]
# ancestors(node, :axis_length, symbol = "A", recursivity_level = 1)
# pipe_model!(node, :cross_section_all, 314, allow_missing = false)
# desc_cross_section = descendants(node, :cross_section, symbol = "S", recursivity_level = 1)