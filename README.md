# Jellyfish: A Tumor Evolution Visualization Tool

Jellyfish automates the creation of Jellyfish plots based on the output from
[ClonEvol](https://doi.org/10.1093/annonc/mdx517) or similar tools that infer
tumor phylogeny and subclonal composition. These plots integrate a hierarchical
sample structure and tumor phylogeny into a single visualization, allowing for
the display of both spatial and temporal evolution of the tumor. The design of
the Jellyfish plot was first introduced in the following paper:

Lahtinen, A., Lavikka, K., Virtanen, A., et al. "Evolutionary states and
trajectories characterized by distinct pathways stratify patients with ovarian
high-grade serous carcinoma." _Cancer Cell_ **41**, 1103–1117.e12 (2023). DOI:
[10.1016/j.ccell.2023.04.017](https://doi.org/10.1016/j.ccell.2023.04.017).

The Jellyfish plots in the paper were drawn manually—a time-consuming and
error-prone process. This tool draws them automatically based on the input data.

You can explore the auto-generated Jellyfish at
[https://hautaniemilab.github.io/jellyfish/](https://hautaniemilab.github.io/jellyfish/),
based on the data from the Lahtinen, et al. (2023) paper, available as example
data in the [`data/`](data/) directory. If you wish to have Jellyfish plots for
your own data, continue reading!

<p align="center">
  <img src="docs/example.svg" alt="An example of a Jellyfish plot" />
</p>

## Basic Concepts

Jellyfish plots visualize the evolution of a tumor by showing the subclonal
composition of samples in a phylogenetic context. The plot combines two trees
into a single visualization: a **phylogeny** and a **sample tree**.

The **phylogeny** is a tree structure that represents the evolutionary
relationships between subclones. Each subclone is a distinct population of cells
with a unique set of genetic mutations.

The samples represent the observed data points, which may be tumor samples from
a patient, each with a unique combination of subclones with specific _clonal
prevalences_, _i.e._ the proportions of the subclones. The **sample tree** is a
tree structure that represents the relationships between samples. The
relationships may be based, for example, on the hypothesized metastatic spread
of the tumor or the chronological order of the samples. Each sample has a rank,
which is a numerical value that determines the position (the column) of the
sample in the plot. The rank can be used to group samples into categories or
time points, such as different stages of a disease. Alternatively, the rank may
be automatically assigned, based on the depth of the sample in the sample tree.

The Jellyfish algorithm optimizes the readability of the visualization by
pushing the emerging subclones towards the leaves of the sample tree. In
practice, Jellyfish finds the [Lowest Common
Ancestor](https://en.wikipedia.org/wiki/Lowest_common_ancestor) (LCA) of each
clade (a subclone and all its descendants) in the sample tree. The LCA subclone
is visualized as an emerging bell, indicating where the subclone first appears
in the sample tree.

Each sample without an explicit parent is considered a child of the _inferred
root_ sample. It is a virtual or hypothetical sample that is used to anchor the
phylogeny to the sample tree, _i.e._, it serves as a host for the LCAs of the
subclones that have been observed in multiple real samples.

## Key Features

- Visualizes tumor phylogeny and subclonal compositions as a Jellyfish plot.
- Allows visualizing both temporal and spatial relationships between samples.
- Sorts samples based on the subclonal composition and divergence, effectively grouping similar samples together.
- Provides basic interactivity for exploring the plot, such as highlighting subclones and clades upon hover or click, and displaying details in tooltips.
- Generates phylogeny-aware color schemes for subclones, inspired by [Visualizing Clonal Evolution in Cancer](http://dx.doi.org/10.1016/j.molcel.2016.05.025) by Krzywinski.
- Exports the plot as publication-ready SVG or PNG files.
- Adjustable layout parameters for fine-tuning the plot appearance.

## Getting Started

If you are an R user, you may want to use the
[Jellyfisher](https://github.com/HautaniemiLab/jellyfisher) R package to
generate Jellyfish plots in RStudio, R Markdown, Shiny apps, or plain R.
Otherwise, continue reading.

### Running with the Development Server

Jellyfish is a web application written in TypeScript. You need to have
[Node.js](https://nodejs.org/) installed to run the tool.

1. `git clone https://github.com/HautaniemiLab/jellyfish.git` (or download the
   repository as a [ZIP archive](https://github.com/HautaniemiLab/jellyfish/archive/refs/heads/main.zip))
2. `cd jellyfish`
3. `npm install`
4. `npm run dev` (starts a development server)

Once the development server is running, open your browser and navigate to
http://localhost:5173/. You should see the user interface, which allows you to
render Jellyfish plots based on your data.

### Building and Deploying as a Static Web Site

If you want to share the interactive Jellyfish plots with others, you can build
the project as an application and deploy it as a static web site on any web
server. An example of such a web site is available at
[https://hautaniemilab.github.io/jellyfish/](https://hautaniemilab.github.io/jellyfish/).

Steps:

1. Perform steps 1-3 from the previous section.
2. `npm run build:app` (builds the project)
3. `cp -R data dist/app/` (copies the example data to the build directory)
4. `cd dist/app`
5. `python3 -m http.server` (starts a local web server for testing)
6. Open your browser and navigate to http://localhost:8000/. You should see the
   user interface.
7. To deploy the site to a web server, copy the contents of the `dist/app`
   directory to the server.

### Building a Jellyfish library

Jellyfish can be used as a library in other JavaScript applications, such as the
[Jellyfisher](https://github.com/HautaniemiLab/jellyfisher) R package. For an
example of how to use the library, see Jellyfisher's [source
code](https://github.com/HautaniemiLab/jellyfisher/blob/main/inst/htmlwidgets/).

Steps:

1. Perform steps 1-3 from the first section.
2. `npm run build:lib` (builds the library)
3. The compiled library is available in the `dist/lib` directory.

## Input Data

Jellyfish reads data as tab-separated files from the `data/` directory. Below is
a description of the data structure, with example files provided in the
directory.

To use your own data, it is recommended to place it in a separate directory,
such as `private-data/`, which is excluded from the Git repository. Then, create
a `.env.local` file (see the Vite
[docs](https://vitejs.dev/guide/env-and-mode.html#env-files) for details) at the
project root with the following content to use the new data directory:

```sh
VITE_DATA_DIR=private-data
```

The structure of the required data files is described below. For datasets
containing a single patient, the `patient` (string) columns can be omitted.

### `samples.tsv`

#### Columns

- `sample` (string): specifies the unique identifier for each sample.
- `displayName` (string, optional): allows for specifying a custom name for each sample. If the column is omitted, the `sample` column is used as the display name.
- `rank` (integer): specifies the position of each sample in the Jellyfish plot. For example, different stages of a disease can be ranked in chronological order: diagnosis (1), interval (2), and relapse (3). The zeroth rank is reserved for the root of the sample tree. Ranks can be any integer, and unused ranks are automatically excluded from the plot. If the `rank` column is
  absent, ranks are assigned based on each sample’s depth in the sample tree.
- `parent` (string): identifies the parent sample for each entry. Samples without a specified parent are treated as children of an imaginary root sample.

#### Example

| sample         | displayName | rank | parent        | patient |
| -------------- | ----------- | ---- | ------------- | ------- |
| P1_iOme_DNA1   | iOme        | 5    |               | P1      |
| P1_iPer1_DNA1  | iPer1       | 5    | P1_pPer1_DNA1 | P1      |
| P1_pAsc_DNA1   | pAsc        | 1    |               | P1      |
| P1_pPer1_DNA1  | pPer1       | 1    |               | P1      |
| P2_iOme2_DNA1  | iOme2       | 5    | P2_pOme2_DNA1 | P2      |
| P2_iOvaR1_DNA1 | iOvaR1      | 5    |               | P2      |
| P2_pOme2_DNA1  | pOme2       | 1    |               | P2      |

### `phylogeny.tsv`

#### Columns

- `subclone` (string): specifies subclone IDs, which can be any string.
- `parent` (string): designates the parent subclone. The subclone without a parent is considered the root of the phylogeny.
- `color` (string, optional): specifies the color for the subclone. If the column is omitted, colors will be generated automatically.
- `branchLength` (number): specifies the length of the branch leading to the subclone. The length may be based on, for example, the number of unique mutations in the subclone. The branch length is shown in the Jellyfish plot's legend as a bar chart. It is also used when generating a phylogeny-aware color scheme.

#### Example

| subclone | parent | color   | branchLength | patient |
| -------- | ------ | ------- | ------------ | ------- |
| 1        |        | #cccccc | 2745         | P1      |
| 2        | 1      | #a6cee3 | 54           | P1      |
| 3        | 1      | #b2df8a | 270          | P1      |
| 5        | 1      | #ff99ff | 216          | P1      |
| 1        |        | #cccccc | 1914         | P2      |
| 4        | 5      | #cab2d6 | 2581         | P2      |
| 5        | 1      | #ff99ff | 1314         | P2      |
| 6        | 1      | #fdbf6f | 1651         | P2      |
| 7        | 6      | #fb9a99 | 137          | P2      |
| 8        | 4      | #bbbb77 | 462          | P2      |

### `compositions.tsv`

Subclonal compositions are specified in a
[tidy](https://vita.had.co.nz/papers/tidy-data.pdf) format, where each row
represents a subclone in a sample.

#### Columns

- `sample` (string): specifies the sample ID.
- `subclone` (string): specifies the subclone ID.
- `clonalPrevalence` (number): specifies the clonal prevalence of the subclone in the sample. The clonal prevalence is the proportion of the subclone in the sample. The clonal prevalences in a sample must sum to 1.

The `sample` and `subclone` columns together form a unique key for each row. The
subclones with no prevalence in a sample are not required to be included in the
table.

#### Example

| sample         | subclone | clonalPrevalence | patient |
| -------------- | -------- | ---------------- | ------- |
| P1_iOme_DNA1   | 1        | 0.842            | P1      |
| P1_iPer1_DNA1  | 1        | 0.78             | P1      |
| P1_pAsc_DNA1   | 1        | 0.174            | P1      |
| P1_pPer1_DNA1  | 2        | 0.874            | P1      |
| P1_iOme_DNA1   | 3        | 0.158            | P1      |
| P1_iPer1_DNA1  | 3        | 0.22             | P1      |
| P1_pAsc_DNA1   | 3        | 0.1655           | P1      |
| P1_pPer1_DNA1  | 3        | 0.125            | P1      |
| P1_pAsc_DNA1   | 5        | 0.6605           | P1      |
| P2_iOme2_DNA1  | 1        | 0.1              | P2      |
| P2_iOvaR1_DNA1 | 1        | 0.024            | P2      |
| P2_pOme2_DNA1  | 1        | 0.1715           | P2      |
| P2_iOme2_DNA1  | 4        | 0.4995           | P2      |
| P2_iOme2_DNA1  | 5        | 0.401            | P2      |
| P2_pOme2_DNA1  | 5        | 0.309            | P2      |
| P2_iOvaR1_DNA1 | 6        | 0.3105           | P2      |
| P2_iOvaR1_DNA1 | 7        | 0.665            | P2      |
| P2_pOme2_DNA1  | 8        | 0.5195           | P2      |

### `ranks.tsv`

Ranks may have optional titles that are displayed above the sample column.

#### Columns

- `rank` (integer): specifies the rank number. The zeroth rank is reserved for the inferred root of the sample tree. However, you are free to define a title for it.
- `title` (string): specifies the title for the rank.

#### Example

| rank | title        |
| ---- | ------------ |
| 0    | Before diag. |
| 1    | Diagnosis    |
| 2    | Diagnosis 2  |
| 3    | Interval     |
| 4    | Relapse      |

## About

Copyright (c) 2024 Kari Lavikka. MIT licensed, see [LICENSE](LICENSE) for details.

Jellyfish is developed in [The Systems Biology of Drug Resistance in
Cancer](https://www.helsinki.fi/en/researchgroups/systems-biology-of-drug-resistance-in-cancer)
group at the [University of Helsinki](https://www.helsinki.fi/en).

This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No. 965193
([DECIDER](https://www.deciderproject.eu/)) and No. 847912
([RESCUER](https://www.rescuer.uio.no/)).
