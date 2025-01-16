import * as d3 from "d3";

export type Subclone = string & { readonly __type: unique symbol };
export type SampleId = string & { readonly __type: unique symbol };

export interface RankRow {
  rank: number;
  title: string;
}

export interface SampleRow {
  sample: SampleId;
  displayName?: string;
  rank: number;
  parent?: SampleId;
  patient?: string;
}

export interface PhylogenyRow {
  subclone: Subclone;
  parent: Subclone;
  color: string;
  branchLength?: number;
  patient?: string;
}

export interface CompositionRow {
  sample: SampleId;
  subclone: Subclone;
  clonalPrevalence: number;
  patient?: string;
}

export interface DataTables {
  ranks?: RankRow[];
  samples: SampleRow[];
  phylogeny: PhylogenyRow[];
  compositions: CompositionRow[];
}

export async function loadDataTables(): Promise<DataTables> {
  const tables = await Promise.all([
    loadAndParseRanks(),
    loadAndParseSamples(),
    loadAndParsePhylogeny(),
    loadAndParseCompositions(),
  ]);
  return {
    ranks: tables[0],
    samples: tables[1],
    phylogeny: tables[2],
    compositions: tables[3],
  };
}

export function filterDataTablesByPatient(
  tables: DataTables,
  patient: string
): DataTables {
  return {
    ranks: tables.ranks,
    samples: tables.samples.filter((d) => d.patient == patient),
    phylogeny: tables.phylogeny.filter((d) => d.patient == patient),
    compositions: tables.compositions.filter((d) => d.patient == patient),
  };
}

async function fetchAndParse(url: string) {
  const dataDir = import.meta.env.VITE_DATA_DIR ?? "data";
  const fullUrl = `${dataDir}/${url}`;

  const response = await fetch(fullUrl);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch ${fullUrl}: ${response.status} ${response.statusText}`
    );
  }
  const text = await response.text();
  return d3.tsvParse(text);
}

export async function loadAndParseRanks(): Promise<RankRow[]> {
  return (await fetchAndParse("ranks.tsv")).map((d) => ({
    rank: parseNumber(d.rank),
    title: d.title,
  }));
}

export async function loadAndParseSamples(): Promise<SampleRow[]> {
  return (await fetchAndParse("samples.tsv")).map((d) => ({
    sample: d.sample as SampleId,
    displayName: d.displayName,
    rank: parseNumber(d.rank),
    parent: d.parent as SampleId,
    patient: d.patient,
  }));
}

export async function loadAndParsePhylogeny(): Promise<PhylogenyRow[]> {
  return (await fetchAndParse("phylogeny.tsv")).map((d) => ({
    subclone: d.subclone as Subclone,
    parent: d.parent as Subclone,
    color: d.color,
    branchLength: parseNumber(d.branchLength),
    patient: d.patient,
  }));
}

export async function loadAndParseCompositions(): Promise<CompositionRow[]> {
  return (await fetchAndParse("compositions.tsv")).map((d) => ({
    sample: d.sample as SampleId,
    subclone: d.subclone as Subclone,
    clonalPrevalence: +d.clonalPrevalence,
    patient: d.patient,
  }));
}

function parseNumber(s: string | undefined) {
  return s !== undefined && s !== "" && s !== "NA" ? +s : undefined;
}
