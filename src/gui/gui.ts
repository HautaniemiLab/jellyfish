import GUI, { Controller } from "lil-gui";
import { DataTables, filterDataTablesByPatient } from "../data.js";
import { tablesToJellyfish } from "../jellyfish.js";
import { LayoutProperties } from "../layout.js";
import { addInteractions } from "../interactions.js";
import { downloadSvg, downloadPng, downloadPdf } from "./download.js";
import { escapeHtml } from "../utils.js";
import { DEFAULT_PROPERTIES } from "../defaultProperties.js";

type ControllerStatus = "open" | "closed" | "hidden";

interface GeneralProperties {
  patient: string | null;
  zoom: number;
}

const DEFAULT_GENERAL_PROPERTIES = {
  patient: null,
  zoom: 1,
} as GeneralProperties;

const ZOOM_EXTENT = [0.2, 3];

export function setupGui(
  container: HTMLElement,
  tables: DataTables,
  customLayoutProps: Partial<LayoutProperties> = {},
  controllerStatus: ControllerStatus = "open"
) {
  container.innerHTML = HTML_TEMPLATE;

  const jellyfishGui = container.querySelector(".jellyfish-gui") as HTMLElement;
  const patientNameElement = jellyfishGui.querySelector(
    ".jellyfish-patient-name"
  ) as HTMLElement;
  const jellyfishPlotContainer = jellyfishGui.querySelector(
    ".jellyfish-plot-container"
  ) as HTMLElement;

  const querySvg = () =>
    jellyfishGui.querySelector(".jellyfish-plot svg") as SVGElement;

  const { generalProps, layoutProps } = getSavedOrDefaultSettings();

  Object.assign(layoutProps, customLayoutProps);

  const saveSettings = () =>
    saveSettingsToSessionStorage(generalProps, layoutProps);

  let translateX = 0;
  let translateY = 0;

  const patients = Array.from(new Set(tables.samples.map((d) => d.patient)));
  if (generalProps.patient && !patients.includes(generalProps.patient)) {
    generalProps.patient = null;
  }
  generalProps.patient ??= patients[0];

  const gui = new GUI({ container: jellyfishGui });
  if (controllerStatus === "closed") {
    gui.close();
  } else if (controllerStatus === "hidden") {
    gui.close();
    gui.hide();
  }

  gui.onChange(saveSettings);

  const isGuiOpen = () => !gui._closed;

  gui.onOpenClose(() => {
    patientNameElement.style.visibility = isGuiOpen() ? "hidden" : null;
  });

  const onPatientChange = () => {
    updatePlot(
      jellyfishGui,
      patients.length > 1
        ? filterDataTablesByPatient(tables, generalProps.patient)
        : tables,
      layoutProps
    );

    generalProps.zoom = 1;
    translateX = 0;
    translateY = 0;

    patientNameElement.textContent = generalProps.patient;

    const svg = querySvg();
    const containerRect = jellyfishPlotContainer.getBoundingClientRect();

    const sufficientZoom = Math.min(
      containerRect.width / +svg.getAttribute("width"),
      containerRect.height / +svg.getAttribute("height")
    );

    generalProps.zoom = Math.min(generalProps.zoom, sufficientZoom);

    onZoomOrPan();
    zoomController.updateDisplay();
  };

  let patientController: Controller;
  if (patients.length > 1) {
    patientController = gui
      .add(generalProps, "patient", patients)
      .onChange(onPatientChange);
  }

  const onZoomOrPan = () => {
    const plot = jellyfishGui.querySelector(".jellyfish-plot") as HTMLElement;
    plot.style.transform = `translate(${translateX}px, ${translateY}px) translate(-50%, -50%) scale(${generalProps.zoom})`;
  };

  const zoomController = gui
    .add(generalProps, "zoom", ZOOM_EXTENT[0], ZOOM_EXTENT[1])
    .onChange(onZoomOrPan);

  const optionFolder = gui.addFolder("Options");
  optionFolder.close();

  const sizeFolder = optionFolder.addFolder("Sizes");
  sizeFolder.add(layoutProps, "sampleHeight", 50, 200);
  sizeFolder.add(layoutProps, "sampleWidth", 50, 200);
  sizeFolder.add(layoutProps, "inferredSampleHeight", 50, 200);
  sizeFolder.add(layoutProps, "gapHeight", 10, 100);
  sizeFolder.add(layoutProps, "sampleSpacing", 10, 200);
  sizeFolder.add(layoutProps, "columnSpacing", 10, 200);
  sizeFolder.add(layoutProps, "sampleFontSize", 8, 16);
  sizeFolder.onChange(onPatientChange);
  sizeFolder.close();

  const bellFolder = optionFolder.addFolder("Bells");
  bellFolder.add(layoutProps, "bellTipShape", 0, 1);
  bellFolder.add(layoutProps, "bellTipSpread", 0, 1);
  bellFolder.add(layoutProps, "bellStrokeWidth", 0, 3);
  bellFolder.add(layoutProps, "bellStrokeDarkening", 0, 2);
  bellFolder.add(layoutProps, "bellPlateauPos", 0.2, 1);
  bellFolder.onChange(onPatientChange);
  bellFolder.close();

  const tentacleFolder = optionFolder.addFolder("Tentacles");
  tentacleFolder.add(layoutProps, "tentacleWidth", 0.1, 5);
  tentacleFolder.add(layoutProps, "tentacleSpacing", 0, 10);
  tentacleFolder.add(layoutProps, "inOutCPDistance", 0.1, 0.45);
  tentacleFolder.add(layoutProps, "bundleCPDistance", 0.1, 1.2);
  tentacleFolder.onChange(onPatientChange);
  tentacleFolder.close();

  const miscFolder = optionFolder.addFolder("Misc.");
  miscFolder.add(layoutProps, "showLegend");
  miscFolder.add(layoutProps, "phylogenyColorScheme");
  miscFolder.add(layoutProps, "phylogenyHueOffset", 0, 360);
  miscFolder.add(layoutProps, "sampleTakenGuide", [
    "none",
    "line",
    "text",
    "text-all",
  ]);
  miscFolder.add(layoutProps, "showRankTitles");
  miscFolder.add(layoutProps, "normalsAtPhylogenyRoot");
  miscFolder.onChange(onPatientChange);
  miscFolder.close();

  const weightsFolder = optionFolder.addFolder("Cost weights");
  weightsFolder.add(layoutProps, "crossingWeight", 0, 10);
  weightsFolder.add(layoutProps, "pathLengthWeight", 0, 10);
  weightsFolder.add(layoutProps, "orderMismatchWeight", 0, 10);
  weightsFolder.add(layoutProps, "divergenceWeight", 0, 10);
  weightsFolder.add(layoutProps, "bundleMismatchWeight", 0, 10);
  weightsFolder.onChange(onPatientChange);
  weightsFolder.close();

  const toolsFolder = gui.addFolder("Tools");
  const tools = {
    downloadSvg: () =>
      downloadSvg(querySvg(), (generalProps.patient ?? "jellyfish") + ".svg"),
    downloadPng: () =>
      downloadPng(querySvg(), (generalProps.patient ?? "jellyfish") + ".png"),
    downloadPdf: () =>
      downloadPdf(querySvg(), (generalProps.patient ?? "jellyfish") + ".pdf"),
  };
  toolsFolder.add(tools, "downloadSvg");
  toolsFolder.add(tools, "downloadPng");
  toolsFolder.add(tools, "downloadPdf");

  if (patientController) {
    setupPatientNavigation(jellyfishGui, patients, generalProps, () => {
      patientController.updateDisplay();
      onPatientChange();
      saveSettings();
    });
  }

  jellyfishPlotContainer.addEventListener("mousedown", (event: MouseEvent) => {
    if (event.button !== 0) {
      return;
    }

    // Allow text selection
    if (["text", "tspan"].includes((event.target as Element).tagName)) {
      return;
    }

    let mouseDownX = event.clientX;
    let mouseDownY = event.clientY;

    const onDrag = (event: MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();

      const dx = event.clientX - mouseDownX;
      const dy = event.clientY - mouseDownY;

      translateX += dx;
      translateY += dy;

      onZoomOrPan();

      mouseDownX = event.clientX;
      mouseDownY = event.clientY;
    };

    container.style.cursor = "grabbing";
    document.addEventListener("mousemove", onDrag);
    container.addEventListener("mouseup", () => {
      document.removeEventListener("mousemove", onDrag);
      container.style.cursor = null;
    });
  });

  jellyfishPlotContainer.addEventListener("wheel", (event: WheelEvent) => {
    event.preventDefault();

    const containerRect = jellyfishPlotContainer.getBoundingClientRect();

    const oldZoom = generalProps.zoom;

    const mouseX = event.clientX - containerRect.left;
    const mouseY = event.clientY - containerRect.top;

    // Coordinates in the plot's coordinate system
    const relativeMouseX =
      (mouseX - containerRect.width / 2 - translateX) / oldZoom;
    const relativeMouseY =
      (mouseY - containerRect.height / 2 - translateY) / oldZoom;

    const newZoom =
      oldZoom *
      2 **
        (-event.deltaY *
          (event.deltaMode === 1 ? 0.05 : event.deltaMode ? 1 : 0.002));
    const clampedZoom = Math.min(
      Math.max(newZoom, ZOOM_EXTENT[0]),
      ZOOM_EXTENT[1]
    );

    generalProps.zoom = clampedZoom;

    translateX -= relativeMouseX * (clampedZoom - oldZoom);
    translateY -= relativeMouseY * (clampedZoom - oldZoom);

    onZoomOrPan();
    zoomController.updateDisplay();
  });

  jellyfishPlotContainer.addEventListener("click", (event: MouseEvent) => {
    // Allow deselection when clicking outside the plot
    if (event.target === jellyfishPlotContainer) {
      querySvg()?.dispatchEvent(new MouseEvent("click", event));
    }
  });

  onZoomOrPan();
  onPatientChange();
}

function updatePlot(
  jellyfishGui: HTMLElement,
  tables: DataTables,
  layoutProps: LayoutProperties
) {
  const plot = jellyfishGui.querySelector(".jellyfish-plot") as HTMLElement;

  try {
    const svg = tablesToJellyfish(tables, layoutProps);
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
      ...DEFAULT_PROPERTIES,
      ...(settings.layoutProps ?? {}),
    } as LayoutProperties,
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
      <div class="jellyfish-center">
        <div class="jellyfish-plot"></div>
      </div>
    </div>
    <div class="jellyfish-patient-nav" style="display: none">
      <button class="jellyfish-prev-patient">
        <svg viewBox="0 0 8 16" fill="currentColor">
          <polygon points="8,0 0,8 8,16" />
        </svg>
        <span>Previous</span>
      </button>
      <div class="jellyfish-patient-name"></div>
      <button class="jellyfish-next-patient">
        <span>Next</span>
        <svg viewBox="0 0 8 16" fill="currentColor">
          <polygon points="0,0 8,8 0,16" />
        </svg>
      </button>
    </div>
  </div>
`;
