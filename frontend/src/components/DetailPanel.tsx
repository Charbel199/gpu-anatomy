import type { AppData, Part } from '../types';
import { DetailHeader } from './DetailHeader';
import { KeyNumbers } from './KeyNumbers';
import { RelatedChips } from './RelatedChips';
import { StatusBadge } from './StatusBadge';

interface Props {
  data: AppData;
  part: Part | null;
  done: boolean;
  onToggleDone: () => void;
  onJump: (id: string) => void;
  codeSection: React.ReactNode;
}

export function DetailPanel({ data, part, done, onToggleDone, onJump, codeSection }: Props) {
  if (!part) {
    return (
      <aside className="detail-panel is-empty">
        <p className="empty-hint">Click a part in the diagram to see its details.</p>
      </aside>
    );
  }

  const mods = part.modules ?? [];
  let codeDone = 0;
  let codePlanned = 0;
  for (const m of mods) {
    codeDone += data.moduleStatus[m]?.done ?? 0;
    codePlanned += data.moduleStatus[m]?.planned ?? 0;
  }

  return (
    <aside className="detail-panel">
      <DetailHeader part={part} />
      <div className="detail-actions">
        <button
          type="button"
          className={'done-toggle' + (done ? ' is-done' : '')}
          onClick={onToggleDone}
          aria-pressed={done}
        >
          {done ? 'Done ✓' : 'Mark as done'}
        </button>
      </div>
      <p className="detail-short">{part.short}</p>
      {part.details && <div className="detail-details">{part.details}</div>}
      <KeyNumbers numbers={part.key_numbers ?? []} />
      <RelatedChips ids={part.related ?? []} partsById={data.partsById} onJump={onJump} />
      <section className="detail-code">
        <div className="section-label">CODE · {mods.length ? mods.join(', ') : '-'}</div>
        {codeSection}
      </section>
      <StatusBadge done={codeDone} total={codePlanned} />
    </aside>
  );
}
