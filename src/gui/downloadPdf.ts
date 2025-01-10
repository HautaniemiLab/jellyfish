import { jsPDF } from "jspdf";
import "svg2pdf.js";

export function downloadPdf(svgElement: SVGElement, filename = "plot.pdf") {
  const width = +svgElement.getAttribute("width");
  const height = +svgElement.getAttribute("height");

  const pdf = new jsPDF({
    unit: "px",
    orientation: width > height ? "l" : "p",
    format: [Math.min(width, height), Math.max(width, height)],
    compress: true,
  });

  pdf
    .svg(svgElement, {
      x: 0,
      y: 0,
      width: svgElement.clientWidth,
      height: svgElement.clientHeight,
    })
    .then(() => {
      pdf.save(filename);
    });
}
