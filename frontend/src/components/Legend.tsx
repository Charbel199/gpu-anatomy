import type { LegendMode } from '../lib/store';

interface Entry {
  key: string;
  label: string;
}

const CATEGORY_ENTRIES: Entry[] = [
  { key: 'cat-scheduling',     label: 'scheduling' },
  { key: 'cat-compute',        label: 'compute' },
  { key: 'cat-memory',         label: 'memory' },
  { key: 'cat-interconnect',   label: 'interconnect' },
  { key: 'cat-fixed-function', label: 'fixed-function' },
  { key: 'cat-special',        label: 'special' },
];

const LATENCY_ENTRIES: Entry[] = [
  { key: 'latency-fast', label: 'fast (~1-5 cycles)' },
  { key: 'latency-mid',  label: 'mid (~30-200 cycles)' },
  { key: 'latency-slow', label: 'slow (~400+ cycles)' },
];

const STATUS_ENTRIES: Entry[] = [
  { key: 'status-done',    label: 'experiments written' },
  { key: 'status-planned', label: 'planned, not yet written' },
];

const MODE_LABEL: Record<LegendMode, string> = {
  category: 'CATEGORY',
  latency: 'LATENCY',
  status: 'STATUS',
};

interface Props {
  mode: LegendMode;
}

export function Legend({ mode }: Props) {
  const entries =
    mode === 'latency' ? LATENCY_ENTRIES :
    mode === 'status'  ? STATUS_ENTRIES  :
                         CATEGORY_ENTRIES;

  return (
    <div className="legend" aria-label="Color legend">
      <span className="legend-mode">{MODE_LABEL[mode]}</span>
      {entries.map((e) => (
        <span key={e.key} className={`legend-item ${e.key}`}>
          <span className="legend-swatch" />
          <span className="legend-label">{e.label}</span>
        </span>
      ))}
    </div>
  );
}
