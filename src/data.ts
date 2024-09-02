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
