const mutedSubcloneFill = "#f8f8f8";
const mutedSubcloneStroke = "#d0d0d0";
const mutedTentacleStroke = "#e8e8e8";

export function addInteractions(svgElement: SVGElement) {
  const originalColors = new Map<
    SVGElement,
    { fill: string; stroke: string }
  >();

  let hoverActive = false;

  for (const elem of svgElement.querySelectorAll(
    ".subclone[data-subclone], .tentacle[data-subclone], .legend-rect[data-subclone]"
  )) {
    originalColors.set(elem as SVGElement, {
      fill: elem.getAttribute("fill"),
      stroke: elem.getAttribute("stroke"),
    });
  }

  svgElement.addEventListener("mouseout", () => {
    if (hoverActive) {
      for (const [elem, { fill, stroke }] of originalColors.entries()) {
        elem.setAttribute("fill", fill);
        elem.setAttribute("stroke", stroke);
      }
      hoverActive = false;
    }
  });

  svgElement.addEventListener("mouseover", (event) => {
    const target = event.target as SVGElement;
    const subclone = target.dataset.subclone;

    if (subclone != "" && subclone != null) {
      for (const [elem, { fill, stroke }] of originalColors.entries()) {
        if (elem.dataset.subclone === subclone) {
          elem.setAttribute("fill", fill);
          elem.setAttribute("stroke", stroke);
        } else {
          if (elem.classList.contains("tentacle")) {
            elem.setAttribute("stroke", mutedTentacleStroke);
          } else {
            elem.setAttribute("fill", mutedSubcloneFill);
            elem.setAttribute("stroke", mutedSubcloneStroke);
          }
        }
      }
      hoverActive = true;
    }
  });
}
