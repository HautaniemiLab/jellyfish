import { G, SVG } from "@svgdotjs/svg.js";
import * as d3 from "d3";
import { Subclone } from "./data.js";
import { drawArrowAndLabel } from "./utilityElements.js";

export interface LegendProperties {
  rectWidth: number;
  rectHeight: number;
  rectSpacing: number;
  fontSize: number;
}

export const DEFAULT_LEGEND_PROPERTIES: LegendProperties = {
  rectWidth: 12,
  rectHeight: 12,
  rectSpacing: 5,
  fontSize: 12,
};

export function getLegendHeight(
  subcloneCount: number,
  props: LegendProperties = DEFAULT_LEGEND_PROPERTIES
) {
  return (
    subcloneCount * (props.rectHeight + props.rectSpacing) - props.rectSpacing
  );
}

export function drawLegend(
  container: G,
  subcloneColors: Map<Subclone, string>,
  branchLengths?: Map<Subclone, number>,
  props: LegendProperties = DEFAULT_LEGEND_PROPERTIES
) {
  const legendGroup = container.group().addClass("legend");

  const entries = Array.from(subcloneColors.entries());

  const getY = (i: number) => i * (props.rectHeight + props.rectSpacing);

  for (let i = 0; i < entries.length; i++) {
    const [subclone, color] = entries[i];

    const y = getY(i);
    const rectangle = legendGroup
      .rect(props.rectWidth, props.rectHeight)
      .fill(color)
      .stroke(d3.color(color).darker(0.6)) // TODO: Configurable darkening
      .move(0, y)
      .addClass("legend-rect")
      .data("subclone", subclone);

    legendGroup
      .text(subclone)
      .font({
        family: "sans-serif",
        size: props.fontSize,
      })
      .attr({ "alignment-baseline": "middle" })
      .move(props.rectWidth + props.rectSpacing, y);

    if (branchLengths) {
      const branchLength = branchLengths.get(subclone);
      if (branchLength) {
        rectangle.add(SVG(`<title>Branch length: ${branchLength}</title>`));
      }
    }
  }

  if (branchLengths) {
    drawArrowAndLabel(
      legendGroup,
      15,
      -17,
      props.rectWidth / 2,
      -props.rectSpacing,
      "Subclone"
    );

    drawBranchLengthGroup(
      legendGroup,
      entries.map(([subclone]) => subclone),
      branchLengths,
      getY,
      props
    );
  }

  return legendGroup;
}

function drawBranchLengthGroup(
  container: G,
  subclones: Subclone[],
  branchLengths: Map<Subclone, number>,
  getY: (i: number) => number,
  props: LegendProperties = DEFAULT_LEGEND_PROPERTIES
) {
  const tickCount = 2;
  const maxOvershoot = 0.2;
  const chartWidth = 40;
  const tickHeight = 4;
  const textRotation = -45;
  const tickStroke = {
    width: 1,
    color: "#a0a0a0",
    linecap: "round",
  };
  const barWidth = 3;

  const lengths = Array.from(branchLengths.values());
  const [minBranchLength, maxBranchLength] = branchLengths
    ? [d3.min, d3.max].map((fn) => fn(lengths))
    : [0, 0];

  const lengthScale = d3
    .scaleLog(
      [
        // Handle cases where there is only a single branch and the min and max
        // are the same.
        minBranchLength < maxBranchLength
          ? minBranchLength
          : maxBranchLength / 10,
        maxBranchLength / (1 + maxOvershoot),
      ],
      [0, chartWidth]
    )
    .nice();

  const lengthGroup = container
    .group()
    .translate(props.rectWidth + props.rectSpacing + 20, 0);

  for (let i = 0; i < subclones.length; i++) {
    const subclone = subclones[i];
    const branchLength = branchLengths?.get(subclone);
    if (branchLength) {
      const y = getY(i);
      const scaledLength = Math.max(1, lengthScale(branchLength));
      const x = 0;
      const x2 = x + scaledLength;
      const cy = y + props.rectHeight / 2;
      lengthGroup.line(x, cy, x2, cy).stroke({
        width: barWidth,
        color: "#b0b0b0",
        linecap: "round",
      });
    }
  }

  const tickGroup = lengthGroup
    .group()
    .addClass("legend-ticks")
    .translate(0, getY(subclones.length));

  const tickFormat = lengthScale.tickFormat(tickCount);

  for (const tick of lengthScale.ticks(tickCount)) {
    const x = lengthScale(tick);
    tickGroup.line(x, 0, x, tickHeight).stroke(tickStroke);

    const tickText = tickFormat(tick);
    if (tickText != "") {
      tickGroup
        .plain(tick.toString())
        .font({
          family: "sans-serif",
          size: props.fontSize * 0.8,
          anchor: "end",
        })
        .attr({
          "alignment-baseline": "middle",
          transform: `translate(${x}, ${
            tickHeight * 2
          }) rotate(${textRotation})`,
        });
    }
  }

  tickGroup.line(0, 0, chartWidth, 0).stroke(tickStroke);

  lengthGroup
    .plain("Variants") // TODO: Configurable label
    .font({
      family: "sans-serif",
      size: 10,
    })
    .translate(-3, 1)
    .addClass("legend-title");

  return lengthGroup;
}
