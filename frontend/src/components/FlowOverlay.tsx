import { useEffect, useState } from 'react';
import type { Flow } from '../types';

interface Props {
  flow: Flow | null;
  onClose: () => void;
  getPartRect: (id: string) => DOMRect | null;
  canvasRect: DOMRect | null;
}

const STEP_MS = 1400;

export function FlowOverlay({ flow, onClose, getPartRect, canvasRect }: Props) {
  const [stepIdx, setStepIdx] = useState(0);

  useEffect(() => {
    if (!flow) return;
    setStepIdx(0);
    const id = window.setInterval(() => {
      setStepIdx((i) => (i + 1) % flow.steps.length);
    }, STEP_MS);
    return () => window.clearInterval(id);
  }, [flow]);

  if (!flow || !canvasRect) return null;

  const step = flow.steps[stepIdx];
  const from = getPartRect(step.from);
  const to = getPartRect(step.to);

  return (
    <div className="flow-overlay">
      <div className="flow-head">
        <span className="flow-title">{flow.title}</span>
        <span className="flow-step">Step {stepIdx + 1}/{flow.steps.length} - {step.label ?? ''}</span>
        <button onClick={onClose}>Close flow</button>
      </div>
      {from && to && (
        <svg
          className="flow-svg"
          style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
          width={canvasRect.width}
          height={canvasRect.height}
        >
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--bp-line)" />
            </marker>
          </defs>
          <line
            x1={from.left - canvasRect.left + from.width / 2}
            y1={from.top  - canvasRect.top  + from.height / 2}
            x2={to.left   - canvasRect.left + to.width / 2}
            y2={to.top    - canvasRect.top  + to.height / 2}
            stroke="var(--bp-line)"
            strokeWidth="2"
            strokeDasharray="4 4"
            markerEnd="url(#arrow)"
          />
        </svg>
      )}
    </div>
  );
}
