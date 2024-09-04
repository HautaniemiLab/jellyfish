# Jellyfish Plotter – a tumor evolution visualization tool

This tool automates the process of creating Jellyfish plots from the output of
ClonEvol or similar tools that infer the phylogeny and subclonality composition
of tumor samples. The Jellyfish visualization design was first introduced in the
following paper: Lahtinen, A. et al. Evolutionary states and trajectories
characterized by distinct pathways stratify patients with ovarian high grade
serous carcinoma. _Cancer Cell_
**41**, 1103–1117.e12 (2023) doi:
[10.1016/j.ccell.2023.04.017](https://doi.org/10.1016/j.ccell.2023.04.017).

The documentation is, obviously, a work in progress.

## Getting started

Jellyfish Plotter is written in JavaScript. You need to have
[Node.js](https://nodejs.org/) installed to run the tool.

1. `git clone git@github.com:HautaniemiLab/jellyfish.git`
2. `cd jellyfish`
3. `npm install`
4. `npm run dev`

## Phases

1. Import data from ClonEvol or some other tool. [#5](https://github.com/HautaniemiLab/jellyfish/issues/5)
2. Based on the subclonal compositions, calculate the pairwise similarities of samples in the same timepoint
3. To reduce redundancy, use the similarities to merge samples that are from the same tissue and are similar enough
4. Construct a sample tree based on the merged samples. [#9](https://github.com/HautaniemiLab/jellyfish/issues/9)
5. Optimize the sample tree by pushing subclones towards the leaves. [#9](https://github.com/HautaniemiLab/jellyfish/issues/9)
6. Use the sample tree, phylogeny and subclonal compositions to generate inferred samples. [#3](https://github.com/HautaniemiLab/jellyfish/issues/3)
7. Compute the placement of the samples by minimizing a cost function that is based on minimizing the lengths and crossings of the tentacles. [#2](https://github.com/HautaniemiLab/jellyfish/issues/2)
8. Render the Jellyfish

The phases should be implemented as separate steps with clearly defined input
and output. This way, the behavior of the tool can be altered or configured by
replacing the phases with alternative implementations or skipping some phases
altogether.

## About

Copyright (c) 2024 Kari Lavikka. See [LICENSE](LICENSE) for details.

Jellyfish Plotter is developed in [The Systems Biology of Drug Resistance in
Cancer](https://www.helsinki.fi/en/researchgroups/systems-biology-of-drug-resistance-in-cancer)
group at the [University of Helsinki](https://www.helsinki.fi/en).

This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No. 965193
([DECIDER](https://www.deciderproject.eu/)) and No. 847912
([RESCUER](https://www.rescuer.uio.no/)).
