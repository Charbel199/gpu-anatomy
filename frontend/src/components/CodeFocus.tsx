import { useEffect, useState } from 'react';
import type { CuFile, Part } from '../types';
import { CodeTabs } from './CodeTabs';

interface Props {
  open: boolean;
  part: Part | null;
  files: CuFile[];
  hoveredPartId: string | null;
  onLineHover: (id: string | null) => void;
  onClose: () => void;
}

const SIZES = [13, 15, 17, 19, 22] as const;

export function CodeFocus({ open, part, files, hoveredPartId, onLineHover, onClose }: Props) {
  const [sizeIdx, setSizeIdx] = useState(1);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); return; }
      if (e.metaKey || e.ctrlKey) return;
      if (e.key === '+' || e.key === '=') {
        e.preventDefault(); setSizeIdx((i) => Math.min(i + 1, SIZES.length - 1));
      } else if (e.key === '-') {
        e.preventDefault(); setSizeIdx((i) => Math.max(i - 1, 0));
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [open, onClose]);

  if (!open) return null;

  const px = SIZES[sizeIdx];

  return (
    <div
      className="codefocus"
      style={{ '--focus-font-size': `${px}px` } as React.CSSProperties}
      onClick={onClose}
    >
      <div className="codefocus-frame" onClick={(e) => e.stopPropagation()}>
        <header className="codefocus-head">
          <div className="codefocus-title">
            {part ? (
              <>
                <span className="codefocus-num">#{part.number}</span>
                <span className={`codefocus-name cat-${part.category}`}>{part.name}</span>
                <span className="codefocus-sub">{(part.modules ?? []).join(', ')}</span>
              </>
            ) : <span>Code</span>}
          </div>
          <div className="codefocus-controls">
            <div className="codefocus-fontsize" role="group" aria-label="Font size">
              <button
                onClick={() => setSizeIdx((i) => Math.max(i - 1, 0))}
                disabled={sizeIdx === 0}
                title="Smaller (-)"
              >A−</button>
              <span className="codefocus-fontval">{px}px</span>
              <button
                onClick={() => setSizeIdx((i) => Math.min(i + 1, SIZES.length - 1))}
                disabled={sizeIdx === SIZES.length - 1}
                title="Larger (+)"
              >A+</button>
            </div>
            <button className="codefocus-close" onClick={onClose} title="Close (Esc)">×</button>
          </div>
        </header>
        <div className="codefocus-body">
          <CodeTabs
            files={files}
            hoveredPartId={hoveredPartId}
            onLineHover={onLineHover}
          />
        </div>
        <footer className="codefocus-foot">
          <span>Esc close</span>
          <span>+ / - resize</span>
        </footer>
      </div>
    </div>
  );
}
