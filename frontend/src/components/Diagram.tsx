import type { Part, View } from '../types';
import type { LegendMode } from '../lib/store';
import { PartBox } from './PartBox';

interface Props {
  view: View;
  partsById: Record<string, Part>;
  selectedId: string | null;
  highlightedIds: Set<string>;
  doneIds: Set<string>;
  legend: LegendMode;
  moduleStatusHas: (partId: string) => boolean;
  fadedIds?: Set<string>;
  onSelect: (partId: string) => void;
  onHover: (partId: string | null) => void;
}

export function Diagram({
  view, partsById, selectedId, highlightedIds, doneIds, legend,
  moduleStatusHas, fadedIds, onSelect, onHover,
}: Props) {
  return (
    <div className={`diagram diagram-${view.layout ?? 'stack'}`}>
      <div className="diagram-title">{view.title.toUpperCase()}</div>
      <div className="diagram-rows" key={view.id}>
        {view.rows.map((row, rIdx) => (
          <div className="diagram-row" key={rIdx}>
            {row.map((partId, cIdx) => {
              const part = partsById[partId];
              if (!part) return null;
              return (
                <PartBox
                  key={`${partId}-${cIdx}`}
                  part={part}
                  legend={legend}
                  hasModules={moduleStatusHas(partId)}
                  done={doneIds.has(partId)}
                  selected={selectedId === partId}
                  highlighted={highlightedIds.has(partId)}
                  faded={fadedIds?.has(partId)}
                  onClick={() => onSelect(partId)}
                  onMouseEnter={() => onHover(partId)}
                  onMouseLeave={() => onHover(null)}
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
