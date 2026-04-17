import type { View, Architecture } from '../types';
import type { LegendMode } from '../lib/store';
import { ViewTabs } from './ViewTabs';
import { ArchSelect } from './ArchSelect';
import { LegendToggle } from './LegendToggle';

interface Props {
  views: View[];
  activeView: string;
  arch: Architecture;
  legend: LegendMode;
  progressDone: number;
  progressTotal: number;
  onArchChange: (a: Architecture) => void;
  onLegendChange: (m: LegendMode) => void;
  onViewChange: (id: string) => void;
  onResetProgress: () => void;
  onOpenHelp: () => void;
}

export function Topbar({
  views, activeView, arch, legend,
  progressDone, progressTotal,
  onArchChange, onLegendChange, onViewChange, onResetProgress, onOpenHelp,
}: Props) {
  const resetTitle = progressDone > 0
    ? `Reset progress (${progressDone} parts marked done)`
    : 'No progress to reset';

  return (
    <header className="topbar">
      <div className="topbar-title">GPU · ANATOMY</div>
      <ViewTabs views={views} activeId={activeView} onChange={onViewChange} />
      <div className="topbar-spacer" />
      <button
        type="button"
        className="progress"
        onClick={() => {
          if (progressDone > 0 && confirm('Reset progress for all parts?')) onResetProgress();
        }}
        title={resetTitle}
        disabled={progressDone === 0}
      >
        {progressDone} / {progressTotal} done
      </button>
      <LegendToggle value={legend} onChange={onLegendChange} />
      <ArchSelect value={arch} onChange={onArchChange} />
      <button
        type="button"
        className="help-btn"
        onClick={onOpenHelp}
        title="Keyboard shortcuts (?)"
        aria-label="Show keyboard shortcuts"
      >?</button>
    </header>
  );
}
