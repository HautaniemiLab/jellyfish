#!/usr/bin/env Rscript

# This script extracts the phylogeny and subclonal compositions from a ClonEvol RDS file.
# DECIDER project uses sample names with a specific format, which incorporates the anatomical
# site and timepoint. Example: X123_p2Per4_DNA1, where X123 is the patient identifier, p2 is the
# timepoint, Per4 is the anatomical site, and DNA1 is the extraction.

suppressMessages({
    library(dplyr)
    library(readr)
    library(stringr)
    library(optparse)
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
  r3 = "Relapse 3"
)

timepoint_df <- data.frame(
  timepoint_code = names(decider_timepoints),
  timepoint = unlist(decider_timepoints),
  stringsAsFactors = FALSE
)
  
args <- OptionParser(
  usage = "%prog [options] clonevol.rds",
  option_list = list(
    make_option("--patient", type="character", default=NA,
                help="The patient indentifier, will be prepended to output file names. Default: NA."),
    make_option("--model", type="integer", default=1,
                help="Which model to use from the clonevol output"),
    make_option("--output", type="character", default=".",
                help="Output directory for the TSV files. Default: \".\"."))
  ) |> parse_args(positional_arguments = 1)


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
    filter(frac > 0.00001) |>
    transmute(sample,
              subclone = cluster,
              proportion = round(frac, 4))
  
  phylogeny <- tree |>
    transmute(subclone = as.integer(lab),
              parent = as.integer(parent),
              color) |>
    arrange(subclone)
  
  
  
  samples <- data.frame(
    sample = subclonal_compositions |>
      pull(sample) |>
      unique()
  ) |>
    mutate(
      m = str_match(sample, "^[\\w\\d]+_(([piro]\\d?)([A-Za-z]+)(\\d)?)"),
      displayName = m[, 2],
      site = m[, 4],
      timepoint_code = m[, 3]
    ) |>
    left_join(timepoint_df, by = join_by(timepoint_code)) |>
    select(-m, -timepoint_code)
  
  return(list(
    samples = samples,
    phylogeny = phylogeny,
    subclonal_compositions = subclonal_compositions
  ))
}

options <- args$options

rds_filename <- args$args[1]
y <- readRDS(rds_filename)
tree <- y$matched$merged.trees[[options$model]]

stopifnot(!is.null(tree))

tables <- extract_tables(tree)

prefix <- if (is.na(options$patient)) "" else paste0(options$patient, "-")

write_tsv(tables$samples, file.path(options$output, paste0(prefix, "samples.tsv")))
write_tsv(tables$phylogeny, file.path(options$output, paste0(prefix, "phylogeny.tsv")))
write_tsv(tables$subclonal_compositions, file.path(options$output, paste0(prefix, "subclonal-compositions.tsv")))