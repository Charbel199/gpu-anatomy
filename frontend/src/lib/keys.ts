import type { View } from '../types';

type Dir = 'left' | 'right' | 'up' | 'down';

export function navigate(view: View, currentId: string | null, dir: Dir): string | null {
  if (!view.rows.length) return null;
  if (!currentId) return view.rows[0]?.[0] ?? null;

  let r = -1, c = -1;
  for (let i = 0; i < view.rows.length; i++) {
    const j = view.rows[i].indexOf(currentId);
    if (j >= 0) { r = i; c = j; break; }
  }
  if (r < 0) return view.rows[0]?.[0] ?? null;

  const row = view.rows[r];
  if (dir === 'left') {
    for (let j = c - 1; j >= 0; j--) if (row[j] !== currentId) return row[j];
  }
  if (dir === 'right') {
    for (let j = c + 1; j < row.length; j++) if (row[j] !== currentId) return row[j];
  }
  if (dir === 'up') {
    for (let i = r - 1; i >= 0; i--) {
      if (view.rows[i].length) return view.rows[i][Math.min(c, view.rows[i].length - 1)];
    }
  }
  if (dir === 'down') {
    for (let i = r + 1; i < view.rows.length; i++) {
      if (view.rows[i].length) return view.rows[i][Math.min(c, view.rows[i].length - 1)];
    }
  }
  return null;
}
