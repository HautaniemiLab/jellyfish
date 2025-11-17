export function downloadPng(svg: SVGElement, filename = "plot.png", dpr = 4) {
  const width = svg.clientWidth * dpr;
  const height = svg.clientHeight * dpr;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const img = new Image();

  const svgBlob = new Blob([svg.outerHTML], {
    type: "image/svg+xml;charset=utf-8",
  });
  const url = URL.createObjectURL(svgBlob);

  img.onload = () => {
    // Draw the SVG image on the canvas
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);
    }

    // Trigger download of the canvas as a PNG
    canvas.toBlob((blob) => {
      if (blob) {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }
    }, "image/png");
  };

  img.src = url;
}

export function downloadSvg(svgElement: SVGElement, filename = "plot.svg") {
  let svgMarkup = svgElement.outerHTML;

  if (!svgMarkup.match(/^<svg[^>]+xmlns="http:\/\/www\.w3\.org\/2000\/svg"/)) {
    svgMarkup = svgMarkup.replace(
      /^<svg/,
      '<svg xmlns="http://www.w3.org/2000/svg"'
    );
  }
  if (!svgMarkup.match(/^<svg[^>]+"http:\/\/www\.w3\.org\/1999\/xlink"/)) {
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

export async function downloadPdf(
  svgElement: SVGElement,
  filename = "plot.pdf"
) {
  // Allow code splitting
  const download = (await import("./downloadPdf.js")).downloadPdf;
  download(svgElement, filename);
}
