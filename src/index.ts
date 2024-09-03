import GUI from "lil-gui";
import { getProportionsBySamples } from "./bellplot.js";
import {
  DataTables,
  filterDataTablesByPatient,
  loadDataTables,
} from "./data.js";
import {
  createBellPlotSvg,
  createBellPlotTreesAndShapers,
  findNodesBySubclone,
} from "./jellyfish.js";
import {
  LayoutProperties,
  optimizeColumns,
  sampleTreeToColumns,
} from "./layout.js";
import { createSampleTreeFromData } from "./sampleTree.js";

const generalProps = {
  patient: null as string,
  zoom: 1,
};

const layoutProps = {
  sampleHeight: 110,
  sampleWidth: 90,
  inferredSampleHeight: 120,
  gapHeight: 60,
  sampleSpacing: 60,
  columnSpacing: 90,
  tentacleSpacing: 5,
  bellTipShape: 0.1,
  bellTipSpread: 0.5,
} as LayoutProperties;

export default async function main() {
  const tables = await loadDataTables();

  const patients = [...new Set(tables.samples.map((d) => d.patient))];
  generalProps.patient ??= "H016"; // patients[0];

  const onPatientChange = () =>
    updatePlot(
      patients.length
        ? filterDataTablesByPatient(tables, generalProps.patient)
        : tables
    );

  const gui = new GUI();
  if (patients.length) {
    gui.add(generalProps, "patient", patients).onChange(onPatientChange);
  }
  gui
    .add(generalProps, "zoom", 0.2, 2)
    .onChange(
      (value: number) =>
        (document.getElementById("plot").style.transform = `scale(${value})`)
    );

  const layoutFolder = gui.addFolder("Layout");
  layoutFolder.add(layoutProps, "sampleHeight", 50, 200);
  layoutFolder.add(layoutProps, "sampleWidth", 50, 200);
  layoutFolder.add(layoutProps, "inferredSampleHeight", 50, 200);
  layoutFolder.add(layoutProps, "gapHeight", 10, 100);
  layoutFolder.add(layoutProps, "sampleSpacing", 10, 200);
  layoutFolder.add(layoutProps, "columnSpacing", 10, 200);
  layoutFolder.add(layoutProps, "tentacleSpacing", 0, 10);
  layoutFolder.add(layoutProps, "bellTipShape", 0, 1);
  layoutFolder.add(layoutProps, "bellTipSpread", 0, 1);

  layoutFolder.onChange(onPatientChange);

  onPatientChange();
}

function updatePlot(tables: DataTables) {
  const { ranks, samples, phylogeny, compositions } = tables;

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
    subcloneLCAs,
    layoutProps
  );

  const subcloneColors = new Map(phylogeny.map((d) => [d.subclone, d.color]));

  const svg = createBellPlotSvg(
    stackedColumns,
    treesAndShapers,
    subcloneColors,
    layoutProps
  );

  const plot = document.getElementById("plot");
  plot.innerHTML = ""; // Purge the old plot

  svg.addTo(plot);
}

main();
