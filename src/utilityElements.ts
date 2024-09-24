import { G } from "@svgdotjs/svg.js";
import * as d3 from "d3";

export function drawArrowAndLabel(
  x: number,
  y: number,
  x2: number,
  y2: number,
  label: string
) {
  const g = new G().addClass("sample-taken-label");

  const strokeWidth = 1;
  const arrowWidth = 3;

  const dir = x2 > x ? 1 : -1;

  // Draws an arrow from (x, y) to (x2, y2)
  // The line is first horizontal, then there's an arc, and then a vertical line,
  // which ends as an arrowhead.

  const p = d3.pathRound(1);

  // Horizontal line
  p.moveTo(x, y);
  p.lineTo(x2 - arrowWidth * dir, y);

  // Arc
  p.arcTo(x2, y, x2, y2, arrowWidth);

  // Vertical line
  p.lineTo(x2, y2);

  g.path(p.toString())
    .addClass("arrow")
    .stroke({ color: "#606060", width: strokeWidth, linecap: "round" })
    .fill("none")
    .attr("marker-end", "url(#small-arrowhead)");

  g.plain(label)
    .font({
      family: "sans-serif",
      size: 10,
      anchor: dir == 1 ? "end" : "start",
    })
    .attr({
      "alignment-baseline": "middle",
    })
    .translate(x - 3 * dir, y);

  return g;
}
