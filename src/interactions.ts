const mutedSubcloneFill = "#f8f8f8";
const mutedSubcloneStroke = "#d0d0d0";
const mutedTentacleStroke = "#e8e8e8";

type State = "normal" | "hover" | "clicked";

export function addInteractions(svgElement: SVGElement) {
  const originalColors = new Map<
    SVGElement,
    { fill: string; stroke: string }
  >();

  let state: State = "normal";

  for (const elem of Array.from(
    svgElement.querySelectorAll(
      ":is(.subclone, .tentacle, .legend-rect, .pass-through)[data-subclone]"
    )
  )) {
    originalColors.set(elem as SVGElement, {
      fill: elem.getAttribute("fill"),
      stroke: elem.getAttribute("stroke"),
    });
  }

  const applyHighlight = (subclones: Set<string> = new Set()) => {
    for (const [elem, { fill, stroke }] of originalColors.entries()) {
      if (subclones.has(elem.dataset.subclone) || subclones.size === 0) {
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
  };

  svgElement.addEventListener("mouseout", () => {
    if (state == "hover") {
      applyHighlight();
    }
  });

  svgElement.addEventListener("mouseover", (event) => {
    if (state === "normal" || state === "hover") {
      const target = event.target as SVGElement;
      const subclones = getSubclones(target.dataset);
      applyHighlight(subclones);

      if (subclones.size > 0) {
        state = "hover";
      }
    }
  });

  svgElement.addEventListener("click", (event) => {
    const target = event.target as SVGElement;
    const subclones = getSubclones(target.dataset);
    applyHighlight(subclones);

    state = subclones.size > 0 ? "clicked" : "normal";
  });
}

function isTentacle(elem: SVGElement) {
  return (
    elem.classList.contains("tentacle") ||
    elem.classList.contains("pass-through")
  );
}

function getSubclones(dataset: DOMStringMap): Set<string> {
  if (dataset.descendantSubclones) {
    return new Set(
      JSON.parse(dataset.descendantSubclones).map(
        (subclone: string | number) => `${subclone}`
      )
    );
  } else if (dataset.subclone != null) {
    return new Set([dataset.subclone]);
  }

  return new Set();
}
