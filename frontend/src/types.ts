export type Category =
  | 'scheduling'
  | 'compute'
  | 'memory'
  | 'interconnect'
  | 'fixed-function'
  | 'special';

export type Architecture =
  | 'Fermi'
  | 'Kepler'
  | 'Maxwell'
  | 'Pascal'
  | 'Volta'
  | 'Turing'
  | 'Ampere'
  | 'Hopper'
  | 'Blackwell';

export interface PartCount {
  per: string;
  n: number;
}

export type LatencyBucket = 'fast' | 'mid' | 'slow';

export interface Part {
  id: string;
  number: number;
  name: string;
  category: Category;
  views: string[];
  parent?: string;
  count?: PartCount;
  short: string;
  key_numbers?: string[];
  details?: string;
  related?: string[];
  modules?: string[];
  latency?: LatencyBucket;
  arch_introduced?: Architecture;
  arch_changes?: Partial<Record<Architecture, string>>;
}

export type ViewLayout = 'stack' | 'pyramid';

export interface View {
  id: string;
  title: string;
  layout?: ViewLayout;
  rows: string[][];
}

export interface FlowStep {
  from: string;
  to: string;
  label?: string;
}

export interface Flow {
  id: string;
  title: string;
  view: string;
  steps: FlowStep[];
}

export interface CuFile {
  path: string;
  folder: string;
  name: string;
  source: string;
  highlighted: string;
}

export interface ModuleStatus {
  folder: string;
  planned: number;
  done: number;
}

export interface AppData {
  parts: Part[];
  views: View[];
  flows: Flow[];
  cuFiles: CuFile[];
  partsById: Record<string, Part>;
  viewsById: Record<string, View>;
  moduleStatus: Record<string, ModuleStatus>;
}
