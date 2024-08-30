#!/usr/bin/env Rscript

# This script extracts the phylogeny and subclonal compositions from a ClonEvol RDS file.
# DECIDER project uses sample names with a specific format, which incorporates the anatomical
# site and timepoint. Example: X123_p2Per4_DNA1, where X123 is the patient identifier, p2 is the
# timepoint, Per4 is the anatomical site, and DNA1 is the extraction.

source("decider_clonevol_to_tsvs.R")

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