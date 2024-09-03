import { getProportionsBySamples } from "./bellplot.js";
import { filterDataTablesByPatient, loadDataTables } from "./data.js";
import {
  createBellPlotSvg,
  createBellPlotTreesAndShapers,
  findNodesBySubclone,
} from "./jellyfish.js";
import {
  columnsToSvg,
  LayoutProperties,
  optimizeColumns,
  sampleTreeToColumns,
} from "./layout.js";
import { createSampleTreeFromData } from "./sampleTree.js";

const layoutProps = {
  sampleHeight: 110,
  sampleWidth: 90,
  inferredSampleHeight: 120,
  gapHeight: 60,
  sampleSpacing: 60,
  columnSpacing: 90,
  canvasWidth: 800,
  canvasHeight: 600,
} as LayoutProperties;

export default async function main() {
  const tables = await loadDataTables();

  const { ranks, samples, phylogeny, compositions } = filterDataTablesByPatient(
    tables,
    "H016"
  );

  const sampleTree = createSampleTreeFromData(samples, ranks);

  const nodesInColumns = sampleTreeToColumns(sampleTree);
  const { stackedColumns, cost } = optimizeColumns(nodesInColumns, layoutProps);

  const proportionsBySamples = getProportionsBySamples(compositions);

  const allSubclones = [...proportionsBySamples.values().next().value.keys()];

  const nodesBySubclone = new Map(
    allSubclones.map((subclone) => [
      subclone,
      findNodesBySubclone(sampleTree, proportionsBySamples, subclone),
    ])
  );

  const subcloneLCAs = new Map(
    [...nodesBySubclone.entries()].map(([subclone, nodes]) => [
      subclone,
      nodes.at(-1),
    ])
  );

  const treesAndShapers = createBellPlotTreesAndShapers(
    sampleTree,
    proportionsBySamples,
    phylogeny,
    allSubclones,
    subcloneLCAs
  );

  const subcloneColors = new Map(phylogeny.map((d) => [d.subclone, d.color]));

  const svg = createBellPlotSvg(
    stackedColumns,
    treesAndShapers,
    subcloneColors,
    layoutProps
  );
  svg.addTo("#app");
}

main();
