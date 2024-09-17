suppressMessages({
  library(dplyr)
  library(readr)
  library(stringr)
})

decider_timepoints <- list(
  p = "Diagnosis",
  p1 = "Diagnosis",
  p2 = "Diagnosis 2",
  p3 = "Diagnosis 3",
  i = "Interval",
  i1 = "Interval",
  i2 = "Interval 2",
  i3 = "Interval 3",
  r = "Relapse",
  r1 = "Relapse",
  r2 = "Relapse 2",
  r3 = "Relapse 3",
  r4 = "Relapse 4"
)

timepoint_df <- data.frame(
  timepoint_code = names(decider_timepoints),
  timepoint = unlist(decider_timepoints),
  stringsAsFactors = FALSE
)
  
extract_tables <- function(tree) {
  # Split the cluster table into separate sample-specific proportions
  all <- matrix(ncol = 4)
  colnames(all) <- c("cluster", "sample", "lower", "upper")
  for (cluster in tree$lab) {
    fracs <- tree |>
      filter(lab == cluster) |>
      pull(sample.with.cell.frac.ci)
    
    samples <- str_split(fracs, ",")[[1]]
    matched <- cbind(cluster,
                     matrix(str_match(samples, "^[°*]?([A-Za-z0-9_]+).* : (-?[0-9.]+)-([0-9.]+)")[,(2:4)], ncol=3))
    
    all <- rbind(all, matched)
  }
  
  subclonal_compositions <- as.data.frame(all) |>
    filter(!is.na(cluster)) |>
    mutate(cluster = as.integer(cluster),
           lower = as.numeric(lower) / 100,
           upper = as.numeric(upper) / 100,
           frac = (lower + upper) / 2) |>
    arrange(cluster, sample) |>
    # 2% is some limit invented by Jaana
    filter(frac > 0.02) |>
    transmute(sample,
              subclone = cluster,
              clonalPrevalence = round(frac, 4))
  
  phylogeny <- tree |>
    transmute(subclone = as.integer(lab),
              parent = as.integer(parent),
              color,
              branchLength = blengths) |>
    arrange(subclone)
  
  samples <- data.frame(
    sample = subclonal_compositions |>
      pull(sample) |>
      unique()
  ) |>
    mutate(
      m = str_match(sample, "^[\\w\\d]+_((([piro]\\d?)([A-Za-z]+)(\\d)?)(_(?!DNA)[A-Za-z\\d]+)?)"),
      displayName = m[, 2],
      site = m[, 5],
      timepoint_code = m[, 4]
    ) |>
    left_join(timepoint_df, by = join_by(timepoint_code)) |>
    select(-m, -timepoint_code)
  
  return(list(
    samples = samples,
    phylogeny = phylogeny,
    subclonal_compositions = subclonal_compositions
  ))
}
