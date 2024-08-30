export type Subclone = string;

export interface RankRow {
  timepoint: string;
  rank: number;
}

export interface SampleRow {
  sample: string;
  displayName?: string;
  site: string;
  timepoint: string;
  patient?: string;
}

export interface PhylogenyRow {
  subclone: Subclone;
  parent: string;
  color: string;
  patient?: string;
}

export interface CompositionRow {
  sample: string;
  subclone: Subclone;
  proportion: number;
  patient?: string;
}

export function getDataTables() {
  return {
    ranks: [] as RankRow[],
    samples: [] as SampleRow[],
    phylogeny: [] as PhylogenyRow[],
    compositions: [] as CompositionRow[],
  };
}
