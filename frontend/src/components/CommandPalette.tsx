import { useEffect, useMemo, useRef, useState } from 'react';
import type { AppData } from '../types';

interface Props {
  data: AppData;
  open: boolean;
  onClose: () => void;
  onPickPart: (id: string) => void;
  onPickView: (id: string) => void;
}

interface Entry {
  type: 'part' | 'view' | 'file';
  id: string;
  label: string;
  sub: string;
  onSelect: () => void;
}

// Subsequence match: all chars of q appear in order somewhere in haystack.
function score(e: Entry, q: string): number {
  if (!q) return 0;
  const hay = (e.label + ' ' + e.sub + ' ' + e.id).toLowerCase();
  let pos = 0;
  for (const ch of q) {
    const i = hay.indexOf(ch, pos);
    if (i < 0) return 0;
    pos = i + 1;
  }
  return 1000 - hay.length + (hay.startsWith(q) ? 500 : 0);
}

export function CommandPalette({ data, open, onClose, onPickPart, onPickView }: Props) {
  const [query, setQuery] = useState('');
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const entries: Entry[] = useMemo(() => {
    const parts: Entry[] = data.parts.map((p) => ({
      type: 'part',
      id: p.id,
      label: `#${p.number} ${p.name}`,
      sub: p.category,
      onSelect: () => onPickPart(p.id),
    }));
    const views: Entry[] = data.views.map((v) => ({
      type: 'view',
      id: v.id,
      label: v.title,
      sub: 'view',
      onSelect: () => onPickView(v.id),
    }));
    const files: Entry[] = data.cuFiles.map((f) => {
      const owningPart = data.parts.find((p) => (p.modules ?? []).includes(f.folder));
      return {
        type: 'file' as const,
        id: f.path,
        label: f.path,
        sub: owningPart?.name ?? f.folder,
        onSelect: () => { if (owningPart) onPickPart(owningPart.id); },
      };
    });
    return [...views, ...parts, ...files];
  }, [data, onPickPart, onPickView]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries.slice(0, 40);
    return entries
      .map((e) => ({ e, score: score(e, q) }))
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 40)
      .map((x) => x.e);
  }, [entries, query]);

  useEffect(() => {
    if (open) { setQuery(''); setCursor(0); inputRef.current?.focus(); }
  }, [open]);

  useEffect(() => { setCursor(0); }, [query]);

  if (!open) return null;

  const pick = (entry: Entry) => { entry.onSelect(); onClose(); };

  return (
    <div className="cmdk-backdrop" onClick={onClose}>
      <div className="cmdk-panel" onClick={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          className="cmdk-input"
          placeholder="Search parts, views, files…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') onClose();
            if (e.key === 'ArrowDown') { e.preventDefault(); setCursor((c) => Math.min(c + 1, filtered.length - 1)); }
            if (e.key === 'ArrowUp')   { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)); }
            if (e.key === 'Enter' && filtered[cursor]) { e.preventDefault(); pick(filtered[cursor]); }
          }}
        />
        <ul className="cmdk-list" role="listbox">
          {filtered.map((entry, i) => (
            <li
              key={entry.type + ':' + entry.id}
              className={'cmdk-row ' + (i === cursor ? 'is-active' : '')}
              onMouseEnter={() => setCursor(i)}
              onClick={() => pick(entry)}
            >
              <span className={`cmdk-tag cmdk-tag-${entry.type}`}>{entry.type}</span>
              <span className="cmdk-label">{entry.label}</span>
              <span className="cmdk-sub">{entry.sub}</span>
            </li>
          ))}
          {filtered.length === 0 && <li className="cmdk-empty">No matches.</li>}
        </ul>
      </div>
    </div>
  );
}
