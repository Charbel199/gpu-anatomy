import type { Architecture } from '../types';
import { useEffect, useState, useCallback } from 'react';

export type LegendMode = 'category' | 'latency' | 'status';

export interface AppState {
  view: string;
  part: string | null;
  arch: Architecture;
  pinned: string[];
  flow: string | null;
  legend: LegendMode;
  trail: string[];
}

export const defaultState: AppState = {
  view: 'chip',
  part: null,
  arch: 'Hopper',
  pinned: [],
  flow: null,
  legend: 'category',
  trail: [],
};

export function parseURL(params: URLSearchParams): AppState {
  const pinned = params.get('pinned');
  const trail = params.get('trail');
  return {
    view: params.get('view') ?? defaultState.view,
    part: params.get('part'),
    arch: (params.get('arch') as Architecture) ?? defaultState.arch,
    pinned: pinned ? pinned.split(',').filter(Boolean) : [],
    flow: params.get('flow'),
    legend: (params.get('legend') as LegendMode) ?? defaultState.legend,
    trail: trail ? trail.split(',').filter(Boolean) : [],
  };
}

export function toURL(s: AppState): URLSearchParams {
  const p = new URLSearchParams();
  if (s.view !== defaultState.view) p.set('view', s.view);
  if (s.part) p.set('part', s.part);
  if (s.arch !== defaultState.arch) p.set('arch', s.arch);
  if (s.pinned.length) p.set('pinned', s.pinned.join(','));
  if (s.flow) p.set('flow', s.flow);
  if (s.legend !== defaultState.legend) p.set('legend', s.legend);
  if (s.trail.length) p.set('trail', s.trail.join(','));
  return p;
}

export function useAppState(): [AppState, (updater: (s: AppState) => AppState) => void] {
  const [state, setState] = useState<AppState>(() =>
    parseURL(new URLSearchParams(window.location.search)),
  );

  useEffect(() => {
    const onPop = () => setState(parseURL(new URLSearchParams(window.location.search)));
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const update = useCallback((updater: (s: AppState) => AppState) => {
    setState((prev) => {
      const next = updater(prev);
      const qs = toURL(next).toString();
      window.history.pushState({}, '', qs ? `?${qs}` : window.location.pathname);
      return next;
    });
  }, []);

  return [state, update];
}
