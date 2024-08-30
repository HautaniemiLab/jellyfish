# Jellyfish – a tumor evolution visualization tool

The documentation is, obviously, a work in progress.

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