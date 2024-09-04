import GUI, { Controller } from "lil-gui";
import {
  DataTables,
  filterDataTablesByPatient,
  loadDataTables,
} from "./data.js";
import { tablesToJellyfish } from "./jellyfish.js";
import { LayoutProperties } from "./layout.js";

interface GeneralProperties {
  patient: string | null;
  zoom: number;
}

const DEFAULT_GENERAL_PROPERTIES = {
  patient: null,
  zoom: 1,
} as GeneralProperties;

const DEFAULT_LAYOUT_PROPERTIES = {
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
  const { generalProps, layoutProps } = getSavedOrDefaultSettings();
  const saveSettings = () =>
    saveSettingsToSessionStorage(generalProps, layoutProps);

  const tables = await loadDataTables();

  const patients = Array.from(new Set(tables.samples.map((d) => d.patient)));
  generalProps.patient ??= patients[0];

  const onPatientChange = () =>
    updatePlot(
      patients.length > 1
        ? filterDataTablesByPatient(tables, generalProps.patient)
        : tables,
      layoutProps
    );

  const gui = new GUI();
  gui.onChange(saveSettings);

  let patientController: Controller;
  if (patients.length > 1) {
    patientController = gui
      .add(generalProps, "patient", patients)
      .onChange(onPatientChange);
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

  if (patientController) {
    addPrevNextKeyboardListeners(patients, generalProps, () => {
      patientController.updateDisplay();
      onPatientChange();
      saveSettings();
    });
  }

  onPatientChange();
}

function updatePlot(tables: DataTables, layoutProps: LayoutProperties) {
  const plot = document.getElementById("plot");
  plot.innerHTML = ""; // Purge the old plot

  try {
    const svg = tablesToJellyfish(tables, layoutProps);
    svg.addTo(plot);
  } catch (e) {
    plot.innerHTML = `<div class="error-message">Error: ${
      (e as Error).message
    }</div>`;
    throw e;
  }
}

const STORAGE_KEY = "jellyfish-plotter-settings";

function saveSettingsToSessionStorage(
  generalProps: GeneralProperties,
  layoutProps: LayoutProperties
) {
  sessionStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ generalProps, layoutProps })
  );
}

function getSavedOrDefaultSettings() {
  const settingsJson = sessionStorage.getItem(STORAGE_KEY) ?? "{}";

  const settings = JSON.parse(settingsJson);
  return {
    generalProps: {
      ...DEFAULT_GENERAL_PROPERTIES,
      ...(settings.generalProps ?? {}),
    } as GeneralProperties,
    layoutProps: {
      ...DEFAULT_LAYOUT_PROPERTIES,
      ...(settings.layoutProps ?? {}),
    } as LayoutProperties,
  };
}

function addPrevNextKeyboardListeners(
  samples: string[],
  generalProps: GeneralProperties,
  onUpdate: (sample: string) => void
) {
  const getSampleByOffset = (offset: number) => {
    const currentIndex = samples.indexOf(generalProps.patient);
    const nextIndex = (currentIndex + offset + samples.length) % samples.length;
    return samples[nextIndex];
  };

  document.addEventListener("keydown", (event) => {
    const currentPatient = generalProps.patient;
    let newPatient: string;

    if (event.key === "ArrowLeft") {
      newPatient = getSampleByOffset(-1);
    } else if (event.key === "ArrowRight") {
      newPatient = getSampleByOffset(1);
    }

    if (newPatient && newPatient !== currentPatient) {
      generalProps.patient = newPatient;
      onUpdate(newPatient);
    }
  });
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
