library(alphashape3d)
library(data.table)
# xyz = fread("0-data/2-lidar_processing/1-trees/crowns/Tree_EC15_crown.txt", data.table = FALSE)

paths <- list.files("0-data/0-raw/1-lidar/", pattern = "*.txt", full.names = TRUE)
subsampling <- 1

volumes <- list()
for (path in paths) {
    tree_name <- gsub(".txt", "", basename(path))
    xyz <- fread(path, data.table = FALSE)
    mat <- as.matrix(xyz[seq(1, nrow(xyz), subsampling), 1:3])
    # plot3d(mat)
    # Compute alpha shape
    ashape <- ashape3d(mat, alpha = 0.8)
    # plot(ashape, transparency = 0.1)
    volumes$tree_name <- volume_ashape3d(ashape)
    tetrahedra <- ashape$tetra[ashape$tetra[, 6] == 1, 1:4]
    colnames(tetrahedra) <- c("p1", "p2", "p3", "p4")
    colnames(ashape$x) <- c("x", "y", "z")
    fwrite(tetrahedra, paste0("0-data/4-sfm_processing/1-trees/", tree_name, "_tetrahedra.csv"), scipen = 10)
    fwrite(ashape$x, paste0("0-data/4-sfm_processing/1-trees/", tree_name, "_vertices.csv"))
}

fwrite(volumes, "0-data/4-sfm_processing/1-trees/volumes.csv")
