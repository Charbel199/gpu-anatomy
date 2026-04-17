import type { Part } from '../types';
import type { LegendMode } from '../lib/store';
import { colorClassFor } from '../lib/classes';

interface Props {
  part: Part;
  legend: LegendMode;
  hasModules: boolean;
  done?: boolean;
  selected?: boolean;
  highlighted?: boolean;
  faded?: boolean;
  onClick?: () => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}

export function PartBox({
  part, legend, hasModules, done, selected, highlighted, faded,
  onClick, onMouseEnter, onMouseLeave,
}: Props) {
  const className = [
    'partbox',
    colorClassFor(part, legend, hasModules),
    selected && 'is-selected',
    highlighted && 'is-highlighted',
    faded && 'is-faded',
    done && 'is-done',
  ].filter(Boolean).join(' ');

  return (
    <button
      type="button"
      className={className}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      data-part-id={part.id}
      aria-label={part.name}
      aria-pressed={done}
    >
      <span className="partbox-label">{part.name.toUpperCase()}</span>
      {part.count && <span className="partbox-count">×{part.count.n}</span>}
      {done && <span className="partbox-check" aria-hidden="true">✓</span>}
    </button>
  );
}
