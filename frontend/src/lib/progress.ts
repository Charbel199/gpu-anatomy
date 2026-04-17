import { useCallback, useState } from 'react';

const STORAGE_KEY = 'gpu-anatomy:progress';

function load(): Set<string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw);
    return new Set(Array.isArray(arr) ? arr : []);
  } catch {
    return new Set();
  }
}

function persist(set: Set<string>): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...set]));
  } catch {
    /* out of quota or private mode; skip */
  }
}

export function useProgress() {
  const [done, setDone] = useState<Set<string>>(load);

  const toggle = useCallback((id: string) => {
    setDone((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      persist(next);
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    const empty = new Set<string>();
    setDone(empty);
    persist(empty);
  }, []);

  return { done, toggle, reset };
}
