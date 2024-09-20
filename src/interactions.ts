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
    ":is(.subclone, .tentacle, .legend-rect, .pass-through)[data-subclone]"
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
    const subclones = getSubclones(target.dataset);

    if (subclones.size > 0) {
      for (const [elem, { fill, stroke }] of originalColors.entries()) {
        if (subclones.has(elem.dataset.subclone)) {
          elem.setAttribute("fill", fill);
          elem.setAttribute("stroke", stroke);
          if (isTentacle(elem)) {
            elem.style.mixBlendMode = "normal";
          }
        } else {
          if (isTentacle(elem)) {
            elem.setAttribute("stroke", mutedTentacleStroke);
            // Ensure that the highlighted tentacle is not covered by the muted ones
            elem.style.mixBlendMode = "darken";
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

function isTentacle(elem: SVGElement) {
  return (
    elem.classList.contains("tentacle") ||
    elem.classList.contains("pass-through")
  );
}

function getSubclones(dataset: DOMStringMap) {
  if (dataset.descendantSubclones) {
    return new Set(JSON.parse(dataset.descendantSubclones));
  } else if (dataset.subclone != null) {
    return new Set([dataset.subclone]);
  }

  return new Set();
}
