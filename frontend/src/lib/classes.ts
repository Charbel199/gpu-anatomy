import type { Part } from '../types';
import type { LegendMode } from './store';

export function colorClassFor(part: Part, legend: LegendMode, hasModules: boolean): string {
  if (legend === 'latency') return part.latency ? `latency-${part.latency}` : 'latency-unknown';
  if (legend === 'status')  return hasModules ? 'status-done' : 'status-planned';
  return `cat-${part.category}`;
}
