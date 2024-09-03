import GUI from "lil-gui";
import {
  DataTables,
  filterDataTablesByPatient,
  loadDataTables,
} from "./data.js";
import { tablesToJellyfish } from "./jellyfish.js";
import { LayoutProperties } from "./layout.js";
import { DEFAULT_LEGEND_PROPERTIES } from "./legend.js";

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
  tentacleWidth: 2,
  tentacleSpacing: 5,
  bellTipShape: 0.1,
  bellTipSpread: 0.5,
  sampleFontSize: 12,
  showLegend: true,
} as LayoutProperties;

export default async function main() {
  const tables = await loadDataTables();

  const patients = [...new Set(tables.samples.map((d) => d.patient))];
  generalProps.patient ??= "H024"; // patients[0];

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
  layoutFolder.add(layoutProps, "tentacleWidth", 0.1, 5);
  layoutFolder.add(layoutProps, "tentacleSpacing", 0, 10);
  layoutFolder.add(layoutProps, "bellTipShape", 0, 1);
  layoutFolder.add(layoutProps, "bellTipSpread", 0, 1);
  layoutFolder.add(layoutProps, "sampleFontSize", 8, 16);
  layoutFolder.add(layoutProps, "showLegend");
  layoutFolder.onChange(onPatientChange);
  layoutFolder.close();

  const toolsFolder = gui.addFolder("Tools");
  toolsFolder.add(
    {
      downloadSvg: () =>
        downloadSvg(
          document.getElementById("plot").querySelector("svg"),
          (generalProps.patient ?? "jellyfish") + ".svg"
        ),
    },
    "downloadSvg"
  );

  onPatientChange();
}

function updatePlot(tables: DataTables) {
  const svg = tablesToJellyfish(tables, layoutProps);
  const plot = document.getElementById("plot");
  plot.innerHTML = ""; // Purge the old plot

  svg.addTo(plot);
}

function downloadSvg(svgElement: SVGElement, filename = "plot.svg") {
  let svgMarkup = svgElement.outerHTML;

  if (!svgMarkup.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) {
    svgMarkup = svgMarkup.replace(
      /^<svg/,
      '<svg xmlns="http://www.w3.org/2000/svg"'
    );
  }
  if (!svgMarkup.match(/^<svg[^>]+"http\:\/\/www\.w3\.org\/1999\/xlink"/)) {
    svgMarkup = svgMarkup.replace(
      /^<svg/,
      '<svg xmlns:xlink="http://www.w3.org/1999/xlink"'
    );
  }

  const blob = new Blob([svgMarkup], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  a.remove();
}

main();
