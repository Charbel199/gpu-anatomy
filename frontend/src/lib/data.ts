import type { Part, View, Flow, AppData, CuFile, ModuleStatus } from '../types';

// @ts-expect-error yaml plugin
import rawParts from '../../data/parts.yaml';
// @ts-expect-error yaml plugin
import rawViews from '../../data/views.yaml';
// @ts-expect-error yaml plugin
import rawFlows from '../../data/flows.yaml';
// @ts-expect-error virtual module from cu-loader
import rawCu from 'virtual:cu-files';

// Planned experiment counts per folder (from gpu.md).
const PLANNED: Record<string, number> = {
  '01_global-memory': 6,
  '02-shared-memory': 7,
  '03-registers': 7,
  '04-warp-behavior': 7,
  '05-warp-shuffle': 7,
  '06-L1-L2-cache': 7,
  '07-atomics': 7,
  '08-occupancy': 7,
  '09-tensor-cores': 7,
  '10-async-copy': 6,
  '11-multi-stream': 7,
  '12-memory-bandwidth': 7,
  '13-constant-memory': 5,
  '14-texture-memory': 6,
  '15-local-memory': 6,
  '16-special-function-units': 7,
  '17-fp64-int32-dual-issue': 6,
  '18-rt-cores': 5,
  '19-nvlink-gpu-to-gpu': 6,
  '20-unified-memory': 6,
  '21-thread-block-clusters': 6,
  '22-instruction-cache': 5,
  '23-transformer-engine': 5,
  '24-mig': 5,
};

function indexBy<T extends { id: string }>(arr: T[]): Record<string, T> {
  return Object.fromEntries(arr.map((x) => [x.id, x]));
}

function buildModuleStatus(cuFiles: CuFile[]): Record<string, ModuleStatus> {
  const done: Record<string, number> = {};
  for (const f of cuFiles) done[f.folder] = (done[f.folder] ?? 0) + 1;
  const out: Record<string, ModuleStatus> = {};
  for (const folder of new Set([...Object.keys(PLANNED), ...Object.keys(done)])) {
    out[folder] = {
      folder,
      planned: PLANNED[folder] ?? done[folder] ?? 0,
      done: done[folder] ?? 0,
    };
  }
  return out;
}

let cached: AppData | null = null;

export function loadAppData(): AppData {
  if (cached) return cached;

  const parts = rawParts as Part[];
  const views = rawViews as View[];
  const flows = rawFlows as Flow[];
  const cuFiles = (rawCu as CuFile[]) ?? [];

  cached = {
    parts,
    views,
    flows,
    cuFiles,
    partsById: indexBy(parts),
    viewsById: indexBy(views),
    moduleStatus: buildModuleStatus(cuFiles),
  };
  return cached;
}
