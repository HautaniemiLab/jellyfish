#
# This script creates stacked bar charts of clonevol (sub)clone proportions.
# I'm using them as a starting point in drawing the Jellyfish plots.
#
#install.packages("devtools")
install.packages("tidyverse")
#devtools::install_github("r-lib/svglite", force=TRUE)

library(tidyverse)
#library(svglite)

dfs <- function(parents) {
    order <- integer()

    do_dfs <- function(parents, current) {
        order <<- c(order, current)
        for (child in which(parents == current)) {
            do_dfs(parents, child)
        }
    }

    do_dfs(parents, 1)

    order
}

setwd("./")
outfolder="./data/preproc2024_1"
inputfolder="./data/2024/"
selected_trees <- read_tsv(paste0(inputfolder,"mutTree_selected_models.csv"))

for (i in seq_len(nrow(selected_trees))) {
    patient <- selected_trees$patient[i]
    model <- selected_trees$model[i]

    print(paste(i, patient))

    dir <- str_glue(paste0(inputfolder,"/",patient))

    y_dir_name <- list.files(dir, "vaf_*")

    y <- readRDS(list.files(file.path(dir, y_dir_name), "*_y.rds", full.names = T))

    tree <- y$matched$merged.trees[[model]]
    expanded_parents <- rep(NA, max(as.integer(tree$lab)))
    expanded_parents[as.integer(tree$lab)] <- as.integer(tree$parent)
    variants = y$variants

    trues <- c()
    i <- 0

    for (b in variants$is.driver) {
      if (b == TRUE) {
        trues <- c(trues, i)
      }
      i <- i + 1
    }

    dfs_order <- data.frame(lab = as.character(seq_along(expanded_parents)),
                            dfs.order = NA)
    d <- dfs(expanded_parents)
    dfs_order$dfs.order[d] <- seq_along(d)

    tree <- tree %>%
        right_join(dfs_order)
    #print(tree)
    #write.csv(tree,paste0(patient,"_all.csv"))

    all <- matrix(ncol = 6)
    colnames(all) <- c("cluster", "parent", "color", "sample", "lower", "upper")

    i = 1
    for (cluster in tree$lab) {

        fracs <- tree %>%
            filter(lab == cluster) %>%

            pull(sample.with.cell.frac.ci)
        c=as.integer(cluster)
        # print(tree$parent)
        #parent = str_split(tree$parent," ")[[c]]
        parent = tree$parent[i]
        samples <- str_split(fracs, ",")[[1]]
        color <- tree$color[i]
        matched <- cbind(cluster, parent, color, matrix(str_match(samples, "_(.+)_.+ : (-?[0-9.]+)-([0-9.]+)")[,(2:4)], ncol=3))
        print(tree$color[i])

        #print("parent:")
        #print(tree$parent[i])

        # print(tree$sample)
        # print("lab:")
        # print(tree$lab)
        all <- rbind(all, matched)

        i=i+1
    }


    all_df <- as.data.frame(all) %>%
        filter(!is.na(cluster)) %>%
        mutate(lower = as.numeric(lower)/ 100,
               upper = as.numeric(upper) / 100,
               frac = (lower + upper) / 2) %>%
        # 2% on joku Jaanan raja
        #filter(frac > 0.02) %>%
        inner_join(tree %>% transmute(cluster = lab, dfs.order))

    colors <- tree$color
    names(colors) <- tree$lab
    write.csv(all_df, paste0(outfolder,"/",patient,".csv"))

    #p <- ggplot(all_df, aes(x = sample, y = frac, fill = cluster, group = dfs.order)) +
    #    geom_bar(position = "fill", stat = "identity") +
    #    scale_fill_manual(values = colors) +
    #    theme_classic()
    #ggsave(p, file = paste0(patient, ".svg"))
}


