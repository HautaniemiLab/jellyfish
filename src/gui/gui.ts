import GUI, { Controller } from "lil-gui";
import { DataTables, filterDataTablesByPatient } from "../data.js";
import { tablesToJellyfish } from "../jellyfish.js";
import { CostWeights, LayoutProperties } from "../layout.js";
import { addInteractions } from "../interactions.js";
import { downloadSvg, downloadPng } from "./download.js";
import { escapeHtml } from "../utils.js";
import {
  DEFAULT_COST_WEIGHTS,
  DEFAULT_PROPERTIES,
} from "../defaultProperties.js";

interface GeneralProperties {
  patient: string | null;
  zoom: number;
}

const DEFAULT_GENERAL_PROPERTIES = {
  patient: null,
  zoom: 1,
} as GeneralProperties;

export function setupGui(container: HTMLElement, tables: DataTables) {
  container.innerHTML = HTML_TEMPLATE;
  const jellyfishGui = container.querySelector(".jellyfish-gui") as HTMLElement;

  const { generalProps, layoutProps, costWeights } =
    getSavedOrDefaultSettings();
  const saveSettings = () =>
    saveSettingsToSessionStorage(generalProps, layoutProps, costWeights);

  const patients = Array.from(new Set(tables.samples.map((d) => d.patient)));
  generalProps.patient ??= patients[0];

  const onPatientChange = () =>
    updatePlot(
      jellyfishGui,
      patients.length > 1
        ? filterDataTablesByPatient(tables, generalProps.patient)
        : tables,
      layoutProps,
      costWeights
    );

  const gui = new GUI({ container: jellyfishGui });
  gui.onChange(saveSettings);

  let patientController: Controller;
  if (patients.length > 1) {
    patientController = gui
      .add(generalProps, "patient", patients)
      .onChange(onPatientChange);
  }

  const onZoomChange = (value: number) =>
    ((
      jellyfishGui.querySelector(".jellyfish-plot") as HTMLElement
    ).style.transform = `scale(${value})`);

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
  layoutFolder.add(layoutProps, "inOutCPDistance", 0.1, 0.45);
  layoutFolder.add(layoutProps, "bundleCPDistance", 0.1, 1.2);
  layoutFolder.add(layoutProps, "bellTipShape", 0, 1);
  layoutFolder.add(layoutProps, "bellTipSpread", 0, 1);
  layoutFolder.add(layoutProps, "bellStrokeWidth", 0, 3);
  layoutFolder.add(layoutProps, "bellPlateauPos", 0.2, 1);
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

  const querySvg = () =>
    jellyfishGui.querySelector(".jellyfish-plot svg") as SVGElement;

  const toolsFolder = gui.addFolder("Tools");
  const tools = {
    downloadSvg: () =>
      downloadSvg(querySvg(), (generalProps.patient ?? "jellyfish") + ".svg"),
    downloadPng: () => {
      downloadPng(querySvg(), (generalProps.patient ?? "jellyfish") + ".png");
    },
  };
  toolsFolder.add(tools, "downloadSvg");
  toolsFolder.add(tools, "downloadPng");

  if (patientController) {
    setupPatientNavigation(jellyfishGui, patients, generalProps, () => {
      patientController.updateDisplay();
      onPatientChange();
      saveSettings();
    });
  }

  onZoomChange(generalProps.zoom);
  onPatientChange();
}

function updatePlot(
  jellyfishGui: HTMLElement,
  tables: DataTables,
  layoutProps: LayoutProperties,
  costWeights: CostWeights
) {
  const plot = jellyfishGui.querySelector(".jellyfish-plot") as HTMLElement;

  try {
    const svg = tablesToJellyfish(tables, layoutProps, costWeights);
    plot.innerHTML = ""; // Purge the old plot
    svg.addTo(plot);

    addInteractions(plot.querySelector("svg"));
  } catch (e) {
    showError(jellyfishGui, (e as Error).message);
    throw e;
  }
}

function showError(container: HTMLElement, message: string) {
  container.querySelector(
    ".jellyfish-plot"
  ).innerHTML = `<div class="jellyfish-error-message">${escapeHtml(
    message
  )}</div>`;
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
      ...DEFAULT_PROPERTIES,
      ...(settings.layoutProps ?? {}),
    } as LayoutProperties,
    costWeights: {
      ...DEFAULT_COST_WEIGHTS,
      ...(settings.costWeights ?? {}),
    } as CostWeights,
  };
}

function setupPatientNavigation(
  jellyfishGui: HTMLElement,
  samples: string[],
  generalProps: GeneralProperties,
  onUpdate: (sample: string) => void
) {
  const navigate = makePatientNavigator(samples, generalProps, onUpdate);

  const patientNav = jellyfishGui.querySelector(
    ".jellyfish-patient-nav"
  ) as HTMLElement;

  // It's "display: none" by default
  patientNav.style.display = null;

  patientNav
    .querySelector(".jellyfish-prev-patient")
    .addEventListener("click", () => navigate(-1));
  patientNav
    .querySelector(".jellyfish-next-patient")
    .addEventListener("click", () => navigate(1));

  // If used in the standalone mode, enable keyboard navigation.
  // Otherwise, like when multiple instances are embedded in a page,
  // we don't want to interfere with the page's keyboard navigation.
  if (jellyfishGui?.parentElement.id === "app") {
    document.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") {
        navigate(-1);
      } else if (event.key === "ArrowRight") {
        navigate(1);
      }
    });
  }
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

const HTML_TEMPLATE = `
  <div class="jellyfish-gui">
    <div class="jellyfish-plot-container">
      <div class="jellyfish-plot"></div>
    </div>
    <div class="jellyfish-patient-nav" style="display: none">
      <button class="jellyfish-prev-patient">
        <svg viewBox="0 0 8 16" fill="currentColor">
          <polygon points="8,0 0,8 8,16" />
        </svg>
        <span>Previous</span>
      </button>
      <button class="jellyfish-next-patient">
        <span>Next</span>
        <svg viewBox="0 0 8 16" fill="currentColor">
          <polygon points="0,0 8,8 0,16" />
        </svg>
      </button>
    </div>
  </div>
`;
