import { DataTables, loadDataTables } from "../data.js";
import { escapeHtml } from "../utils.js";
import { setupGui } from "./gui.js";

export default async function main() {
  const appElement = document.getElementById("app");

  let tables: DataTables;
  try {
    tables = await loadDataTables();
  } catch (e) {
    showError(appElement, (e as Error).message);
    throw e;
  }

  setupGui(appElement, tables);
}

function showError(container: HTMLElement, message: string) {
  container.innerHTML = `<div class="jellyfish-error-message">${escapeHtml(
    message
  )}</div>`;
}

main();
