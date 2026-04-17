import type { Part } from '../types';

export function DetailHeader({ part }: { part: Part }) {
  return (
    <header className="detail-header">
      <div className="detail-num">#{part.number}</div>
      <h2 className={`detail-name cat-${part.category}`}>{part.name}</h2>
      <div className="detail-meta">
        {part.count && <span>{part.count.n} per {part.count.per}</span>}
        {part.arch_introduced && <span>since {part.arch_introduced}</span>}
        <span>{part.category}</span>
      </div>
    </header>
  );
}
