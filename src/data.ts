import * as d3 from "d3";

export type Subclone = string & { readonly __type: unique symbol };
export type SampleId = string & { readonly __type: unique symbol };

export interface RankRow {
  timepoint: string;
  rank: number;
}

export interface SampleRow {
  sample: SampleId;
  displayName?: string;
  site: string;
  timepoint: string;
  patient?: string;
}

export interface PhylogenyRow {
  subclone: Subclone;
  parent: Subclone;
  color: string;
  patient?: string;
}

export interface CompositionRow {
  sample: SampleId;
  subclone: Subclone;
  clonalPrevalence: number;
  patient?: string;
}

export interface DataTables {
  ranks: RankRow[];
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
  const text = await fetch(url).then((response) => response.text());
  return d3.tsvParse(text);
}

export async function loadAndParseRanks(): Promise<RankRow[]> {
  return (await fetchAndParse("data/ranks.tsv")).map((d) => ({
    timepoint: d.timepoint,
    rank: +d.rank,
  }));
}

export async function loadAndParseSamples(): Promise<SampleRow[]> {
  return (await fetchAndParse("data/samples.tsv")).map((d) => ({
    sample: d.sample as SampleId,
    displayName: d.displayName,
    site: d.site,
    timepoint: d.timepoint,
    patient: d.patient,
  }));
}

export async function loadAndParsePhylogeny(): Promise<PhylogenyRow[]> {
  return (await fetchAndParse("data/phylogeny.tsv")).map((d) => ({
    subclone: d.subclone as Subclone,
    parent: d.parent as Subclone,
    color: d.color,
    patient: d.patient,
  }));
}

export async function loadAndParseCompositions(): Promise<CompositionRow[]> {
  return (await fetchAndParse("data/subclonal_compositions.tsv")).map((d) => ({
    sample: d.sample as SampleId,
    subclone: d.subclone as Subclone,
    clonalPrevalence: +d.clonalPrevalence,
    patient: d.patient,
  }));
}
