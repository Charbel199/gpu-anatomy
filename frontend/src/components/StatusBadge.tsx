interface Props {
  done: number;
  total: number;
}

export function StatusBadge({ done, total }: Props) {
  if (total === 0) return null;
  const complete = done === total;
  return (
    <div className={'status-badge' + (complete ? ' is-complete' : '')}>
      <span className="status-fraction">{done}/{total}</span>
      <span className="status-mark">{complete ? '✓' : '…'}</span>
      <span className="status-label">{complete ? 'experiments complete' : 'experiments planned'}</span>
    </div>
  );
}
