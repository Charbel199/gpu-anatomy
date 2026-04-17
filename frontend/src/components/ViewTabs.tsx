import type { View } from '../types';

interface Props {
  views: View[];
  activeId: string;
  onChange: (id: string) => void;
}

export function ViewTabs({ views, activeId, onChange }: Props) {
  return (
    <div className="viewtabs" role="tablist">
      {views.map((v, i) => (
        <button
          key={v.id}
          role="tab"
          aria-selected={v.id === activeId}
          className={'viewtab' + (v.id === activeId ? ' is-active' : '')}
          onClick={() => onChange(v.id)}
          title={`${v.title} · press ${i + 1}`}
        >
          <span className="viewtab-num">{i + 1}</span>
          <span className="viewtab-label">{v.id.toUpperCase()}</span>
        </button>
      ))}
    </div>
  );
}
