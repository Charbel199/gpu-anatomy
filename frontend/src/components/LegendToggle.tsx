import type { LegendMode } from '../lib/store';

interface Props { value: LegendMode; onChange: (v: LegendMode) => void; }

const MODES: Array<{ id: LegendMode; label: string }> = [
  { id: 'category', label: 'category' },
  { id: 'latency',  label: 'latency'  },
  { id: 'status',   label: 'status'   },
];

export function LegendToggle({ value, onChange }: Props) {
  return (
    <div className="legend-toggle" role="radiogroup" aria-label="Color mode">
      {MODES.map((m) => (
        <button
          key={m.id}
          role="radio"
          aria-checked={value === m.id}
          className={'legend-opt' + (value === m.id ? ' is-active' : '')}
          onClick={() => onChange(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
