import type { Part } from '../types';

export function computeHighlightSet(
  selectedId: string | null,
  partsById: Record<string, Part>,
): Set<string> {
  if (!selectedId) return new Set();
  const selected = partsById[selectedId];
  if (!selected) return new Set();
  const out = new Set<string>();
  for (const r of selected.related ?? []) {
    if (partsById[r]) out.add(r);
  }
  return out;
}
