import { G } from "@svgdotjs/svg.js";
import * as d3 from "d3";
import { Subclone } from "./data.js";

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

export function drawLegend(
  subcloneColors: Map<Subclone, string>,
  branchLengths?: Map<Subclone, number>,
  props: LegendProperties = DEFAULT_LEGEND_PROPERTIES
) {
  const maxBranchLength = branchLengths
    ? d3.max(Array.from(branchLengths.values()))
    : 0;

  const g = new G();
  g.addClass("legend");

  const totalHeight =
    subcloneColors.size * (props.rectHeight + props.rectSpacing) -
    props.rectSpacing;

  const entries = Array.from(subcloneColors.entries());

  for (let i = 0; i < entries.length; i++) {
    const [subclone, color] = entries[i];

    const y = i * (props.rectHeight + props.rectSpacing) - totalHeight / 2;

    g.rect(props.rectWidth, props.rectHeight)
      .fill(color)
      .stroke(d3.color(color).darker(0.6)) // TODO: Configurable darkening
      .move(0, y);
    g.text(subclone)
      .font({
        family: "sans-serif",
        size: props.fontSize,
      })
      .attr({ "dominant-baseline": "central" })
      .move(props.rectWidth + props.rectSpacing, y);

    const branchLength = branchLengths?.get(subclone);
    if (branchLength) {
      const scaledLength = Math.max(1, (branchLength / maxBranchLength) * 35);
      const x = props.rectWidth + props.rectSpacing + 20;
      const x2 = x + scaledLength;
      const cy = y + props.rectHeight / 2;
      g.line(x, cy, x2, cy).stroke({
        width: 3,
        color: "#a0a0a0",
        linecap: "round",
      });
    }
  }

  return g;
}
