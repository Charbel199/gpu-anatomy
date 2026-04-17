import type { Part } from '../types';

interface Props {
  ids: string[];
  partsById: Record<string, Part>;
  onJump: (id: string) => void;
}

export function RelatedChips({ ids, partsById, onJump }: Props) {
  if (!ids.length) return null;
  return (
    <div className="related">
      <div className="section-label">RELATED</div>
      <ul className="related-list">
        {ids.map((id) => {
          const p = partsById[id];
          if (!p) return null;
          return (
            <li key={id}>
              <button
                className={`chip chip-link cat-${p.category}`}
                onClick={() => onJump(id)}
              >
                {p.name}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
