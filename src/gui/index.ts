import GUI, { Controller } from "lil-gui";
import {
  DataTables,
  filterDataTablesByPatient,
  loadDataTables,
} from "../data.js";
import { tablesToJellyfish } from "../jellyfish.js";
import {
  CostWeights,
  DEFAULT_COST_WEIGHTS,
  LayoutProperties,
} from "../layout.js";
import { addInteractions } from "../interactions.js";
import { downloadSvg, downloadPng } from "./download.js";
import { DEFAULT_BELL_PLOT_PROPERTIES } from "../bellplot.js";

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
  sampleFontSize: 12,
  showLegend: true,
  phylogenyColorScheme: true,
  phylogenyHueOffset: 0,
  sampleTakenGuide: "text",
  ...DEFAULT_BELL_PLOT_PROPERTIES,
} as LayoutProperties;

export default async function main() {
  const { generalProps, layoutProps, costWeights } =
    getSavedOrDefaultSettings();
  const saveSettings = () =>
    saveSettingsToSessionStorage(generalProps, layoutProps, costWeights);

  let tables: DataTables;
  try {
    tables = await loadDataTables();
  } catch (e) {
    showError((e as Error).message);
    throw e;
  }

  const patients = Array.from(new Set(tables.samples.map((d) => d.patient)));
  generalProps.patient ??= patients[0];

  const onPatientChange = () =>
    updatePlot(
      patients.length > 1
        ? filterDataTablesByPatient(tables, generalProps.patient)
        : tables,
      layoutProps,
      costWeights
    );

  const gui = new GUI();
  gui.onChange(saveSettings);

  let patientController: Controller;
  if (patients.length > 1) {
    patientController = gui
      .add(generalProps, "patient", patients)
      .onChange(onPatientChange);
  }

  const onZoomChange = (value: number) =>
    (document.getElementById("plot").style.transform = `scale(${value})`);

  gui.add(generalProps, "zoom", 0.2, 2).onChange(onZoomChange);

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
  layoutFolder.add(layoutProps, "bellStrokeWidth", 0, 3);
  layoutFolder.add(layoutProps, "plateauPos", 0.2, 1);
  layoutFolder.add(layoutProps, "sampleFontSize", 8, 16);
  layoutFolder.add(layoutProps, "showLegend");
  layoutFolder.add(layoutProps, "phylogenyColorScheme");
  layoutFolder.add(layoutProps, "phylogenyHueOffset", 0, 360);
  layoutFolder.add(layoutProps, "sampleTakenGuide", ["none", "line", "text"]);
  layoutFolder.onChange(onPatientChange);
  layoutFolder.close();

  const weightsFolder = gui.addFolder("Cost weights");
  weightsFolder.add(costWeights, "crossing", 0, 10);
  weightsFolder.add(costWeights, "pathLength", 0, 10);
  weightsFolder.add(costWeights, "orderMismatch", 0, 10);
  weightsFolder.add(costWeights, "divergence", 0, 10);
  weightsFolder.add(costWeights, "bundleMismatch", 0, 10);
  weightsFolder.onChange(onPatientChange);
  weightsFolder.close();

  const toolsFolder = gui.addFolder("Tools");
  const tools = {
    downloadSvg: () =>
      downloadSvg(
        document.getElementById("plot").querySelector("svg"),
        (generalProps.patient ?? "jellyfish") + ".svg"
      ),
    downloadPng: () => {
      downloadPng(
        document.getElementById("plot").querySelector("svg"),
        (generalProps.patient ?? "jellyfish") + ".png"
      );
    },
  };
  toolsFolder.add(tools, "downloadSvg");
  toolsFolder.add(tools, "downloadPng");

  if (patientController) {
    setupPatientNavigation(patients, generalProps, () => {
      patientController.updateDisplay();
      onPatientChange();
      saveSettings();
    });
  }

  onZoomChange(generalProps.zoom);
  onPatientChange();
}

function updatePlot(
  tables: DataTables,
  layoutProps: LayoutProperties,
  costWeights: CostWeights
) {
  const plot = document.getElementById("plot");

  try {
    const svg = tablesToJellyfish(tables, layoutProps, costWeights);
    plot.innerHTML = ""; // Purge the old plot
    svg.addTo(plot);

    addInteractions(plot.querySelector("svg"));
  } catch (e) {
    showError((e as Error).message);
    throw e;
  }
}

function showError(message: string) {
  document.getElementById(
    "plot"
  ).innerHTML = `<div class="error-message">${message}</div>`;
}

const STORAGE_KEY = "jellyfish-plotter-settings";

function saveSettingsToSessionStorage(
  generalProps: GeneralProperties,
  layoutProps: LayoutProperties,
  costWeights: CostWeights
) {
  sessionStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ generalProps, layoutProps, costWeights })
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
    costWeights: {
      ...DEFAULT_COST_WEIGHTS,
      ...(settings.costWeights ?? {}),
    } as CostWeights,
  };
}

function setupPatientNavigation(
  samples: string[],
  generalProps: GeneralProperties,
  onUpdate: (sample: string) => void
) {
  const navigate = makePatientNavigator(samples, generalProps, onUpdate);

  // It's "display: none" by default
  document.getElementById("patient-nav").style.display = null;

  document
    .getElementById("prev-patient")
    .addEventListener("click", () => navigate(-1));
  document
    .getElementById("next-patient")
    .addEventListener("click", () => navigate(1));

  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      navigate(-1);
    } else if (event.key === "ArrowRight") {
      navigate(1);
    }
  });
}

function makePatientNavigator(
  samples: string[],
  generalProps: GeneralProperties,
  onUpdate: (sample: string) => void
) {
  const getPatientByOffset = (offset: number) => {
    const currentIndex = samples.indexOf(generalProps.patient);
    const nextIndex = (currentIndex + offset + samples.length) % samples.length;
    return samples[nextIndex];
  };

  return function navigatePatients(direction: number) {
    const currentPatient = generalProps.patient;
    const newPatient = getPatientByOffset(direction);

    if (newPatient && newPatient !== currentPatient) {
      generalProps.patient = newPatient;
      onUpdate(newPatient);
    }
  };
}

main();
