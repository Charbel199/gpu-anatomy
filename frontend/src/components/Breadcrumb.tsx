import type { View } from '../types';

interface Props {
  trail: string[];
  currentView: View;
  viewsById: Record<string, View>;
  onClimb: (level: number) => void;
}

function shortTitle(title: string): string {
  return title.split('-')[0].trim();
}

export function Breadcrumb({ trail, currentView, viewsById, onClimb }: Props) {
  if (trail.length === 0) return null;
  return (
    <nav className="breadcrumb" aria-label="Drill-in path">
      {trail.map((vid, i) => {
        const v = viewsById[vid];
        if (!v) return null;
        return (
          <span key={i}>
            <button className="crumb" onClick={() => onClimb(i)}>{shortTitle(v.title)}</button>
            <span className="crumb-sep">›</span>
          </span>
        );
      })}
      <span className="crumb is-current">{shortTitle(currentView.title)}</span>
    </nav>
  );
}
