import { useEffect, useState } from 'react';
import type { Part } from '../types';

interface Props {
  part: Part | null;
  legendClass: string;
}

export function Tooltip({ part, legendClass }: Props) {
  const [pos, setPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const onMove = (e: MouseEvent) => setPos({ x: e.clientX, y: e.clientY });
    window.addEventListener('mousemove', onMove);
    return () => window.removeEventListener('mousemove', onMove);
  }, []);

  if (!part) return null;

  return (
    <div
      className="tooltip"
      style={{ transform: `translate(${pos.x + 16}px, ${pos.y + 16}px)` }}
      role="tooltip"
    >
      <div className={`tooltip-name ${legendClass}`}>{part.name}</div>
      {part.count && <div className="tooltip-count">{part.count.n} per {part.count.per}</div>}
      <div className="tooltip-short">{part.short}</div>
    </div>
  );
}
