import { useMemo, useState } from 'react';
import type { CuFile } from '../types';
import { GithubIcon } from './GithubIcon';

interface Props {
  files: CuFile[];
  hoveredPartId: string | null;
  onLineHover: (partId: string | null) => void;
  onExpand?: () => void;
}

const PART_RE = /\/\/\s*@part:\s*([a-z0-9_]+)/;

function parseLineParts(source: string): Array<string | null> {
  return source.split('\n').map((line) => {
    const m = PART_RE.exec(line);
    return m ? m[1] : null;
  });
}

export function CodeTabs({ files, hoveredPartId, onLineHover, onExpand }: Props) {
  const [active, setActive] = useState(0);

  const current: CuFile | null = files.length
    ? files[Math.min(active, files.length - 1)]
    : null;

  const lineParts = useMemo(
    () => (current ? parseLineParts(current.source) : []),
    [current],
  );

  const injected = useMemo(() => {
    if (!current) return '';
    const chunks = current.highlighted.split(/(?=<span class="line")/);
    return chunks.map((chunk, i) => {
      const pid = lineParts[i] ?? null;
      if (!pid) return chunk;
      const ribbon = hoveredPartId && pid === hoveredPartId ? ' data-ribbon="1"' : '';
      return chunk.replace('<span class="line"', `<span class="line" data-part-id="${pid}"${ribbon}`);
    }).join('');
  }, [current, lineParts, hoveredPartId]);

  if (!current) {
    return <div className="codetabs is-empty">No code experiments exist yet for this part.</div>;
  }

  return (
    <div className="codetabs">
      <div className="codetabs-tabs" role="tablist">
        {files.map((f, i) => (
          <button
            key={f.path}
            role="tab"
            aria-selected={i === active}
            className={'codetab' + (i === active ? ' is-active' : '')}
            onClick={() => setActive(i)}
          >
            {f.name}
          </button>
        ))}
      </div>
      <div className="codetabs-path">
        <span>{current.path}</span>
        <span className="codetabs-path-right">
          <span className="codetabs-count">{active + 1} / {files.length}</span>
          <GithubIcon path={current.path} />
          {onExpand && (
            <button
              type="button"
              className="codetabs-expand"
              onClick={onExpand}
              title="Focus mode (bigger code)"
              aria-label="Expand code to focus mode"
            >⤢</button>
          )}
        </span>
      </div>
      <div
        className="codetabs-body"
        onMouseOver={(e) => {
          const line = (e.target as HTMLElement).closest('.line') as HTMLElement | null;
          onLineHover(line?.getAttribute('data-part-id') ?? null);
        }}
        onMouseLeave={() => onLineHover(null)}
        dangerouslySetInnerHTML={{ __html: injected }}
      />
    </div>
  );
}
