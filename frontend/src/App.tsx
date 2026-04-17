import { useEffect, useState, useMemo, useRef, useCallback } from 'react';
import type { AppData } from './types';
import { loadAppData } from './lib/data';
import { useAppState } from './lib/store';
import { isVisibleInArch } from './lib/arch';
import { computeHighlightSet } from './lib/highlight';
import { navigate } from './lib/keys';
import { useProgress } from './lib/progress';
import { colorClassFor } from './lib/classes';
import { Topbar } from './components/Topbar';
import { Diagram } from './components/Diagram';
import { DetailPanel } from './components/DetailPanel';
import { CodeTabs } from './components/CodeTabs';
import { Tooltip } from './components/Tooltip';
import { CommandPalette } from './components/CommandPalette';
import { HelpOverlay } from './components/HelpOverlay';
import { Breadcrumb } from './components/Breadcrumb';
import { PinCompare } from './components/PinCompare';
import { FlowOverlay } from './components/FlowOverlay';
import { CodeFocus } from './components/CodeFocus';
import { Legend } from './components/Legend';

const DIR_MAP: Record<string, 'left' | 'right' | 'up' | 'down'> = {
  ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down',
  h: 'left', l: 'right', k: 'up', j: 'down',
};

export default function App() {
  const [data, setData] = useState<AppData | null>(null);
  const [state, setState] = useAppState();
  const { done: doneIds, toggle: toggleDone, reset: resetProgress } = useProgress();

  const [hoverId, setHoverId] = useState<string | null>(null);
  const [codeHoverId, setCodeHoverId] = useState<string | null>(null);
  const [cmdOpen, setCmdOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [focusOpen, setFocusOpen] = useState(false);
  const diagramRef = useRef<HTMLElement>(null);

  useEffect(() => { setData(loadAppData()); }, []);

  const activeView = useMemo(
    () => (data ? data.viewsById[state.view] ?? data.views[0] ?? null : null),
    [data, state.view],
  );

  const visiblePartsById = useMemo(() => {
    if (!data) return {};
    const out: Record<string, (typeof data.parts)[number]> = {};
    for (const p of data.parts) {
      if (isVisibleInArch(p, state.arch)) out[p.id] = p;
    }
    return out;
  }, [data, state.arch]);

  const selectedPart = useMemo(
    () => (data && state.part ? data.partsById[state.part] ?? null : null),
    [data, state.part],
  );

  const hoverPart = hoverId && data ? data.partsById[hoverId] ?? null : null;

  const codeFiles = useMemo(() => {
    if (!data || !selectedPart) return [];
    const folders = new Set(selectedPart.modules ?? []);
    return data.cuFiles.filter((f) => folders.has(f.folder));
  }, [data, selectedPart]);

  const fadedIds = useMemo(() => {
    if (!data) return new Set<string>();
    const out = new Set<string>();
    for (const p of data.parts) {
      const mods = p.modules ?? [];
      if (mods.length > 0 && mods.every((m) => (data.moduleStatus[m]?.done ?? 0) === 0)) {
        out.add(p.id);
      }
    }
    return out;
  }, [data]);

  const highlightedIds = useMemo(() => {
    const ids = computeHighlightSet(state.part, data?.partsById ?? {});
    if (codeHoverId) ids.add(codeHoverId);
    return ids;
  }, [state.part, data, codeHoverId]);

  const moduleStatusHas = useCallback((partId: string) => {
    if (!data) return false;
    const p = data.partsById[partId];
    return !!p && (p.modules ?? []).some((m) => (data.moduleStatus[m]?.done ?? 0) > 0);
  }, [data]);

  const switchView = useCallback((view: string) => {
    setState((s) => ({ ...s, view, part: null, trail: [] }));
  }, [setState]);

  const selectPart = useCallback((part: string) => {
    setState((s) => ({ ...s, part }));
  }, [setState]);

  const drillInto = useCallback((partId: string) => {
    if (!data?.viewsById[partId]) return;
    setState((s) => ({ ...s, view: partId, trail: [...s.trail, s.view], part: null }));
  }, [data, setState]);

  const climb = useCallback((level: number) => {
    setState((s) => ({
      ...s,
      view: s.trail[level] ?? s.view,
      trail: s.trail.slice(0, level),
      part: null,
    }));
  }, [setState]);

  const togglePin = useCallback((id: string) => {
    setState((s) => {
      const pinned = s.pinned.includes(id)
        ? s.pinned.filter((x) => x !== id)
        : [...s.pinned.slice(-1), id].slice(-2);
      return { ...s, pinned };
    });
  }, [setState]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const inField = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA';
      const mod = e.metaKey || e.ctrlKey;

      if ((mod && e.key.toLowerCase() === 'k') || (e.key === '/' && !inField)) {
        e.preventDefault();
        setCmdOpen(true);
        return;
      }

      if (inField || cmdOpen) return;

      if (/^[1-9]$/.test(e.key)) {
        const v = data?.views[parseInt(e.key, 10) - 1];
        if (v) { e.preventDefault(); switchView(v.id); }
        return;
      }

      if (e.key === '?') { e.preventDefault(); setHelpOpen(true); return; }

      if (e.key === 'Escape') {
        if (focusOpen)  { setFocusOpen(false); return; }
        if (helpOpen)   { setHelpOpen(false);  return; }
        if (state.flow) { setState((s) => ({ ...s, flow: null })); return; }
        if (state.trail.length > 0) { climb(state.trail.length - 1); return; }
        setState((s) => ({ ...s, part: null }));
        return;
      }

      if (e.key === 'Enter' && state.part) {
        e.preventDefault();
        drillInto(state.part);
        return;
      }

      const k = e.key.toLowerCase();

      if (state.part) {
        if (k === 'e' && codeFiles.length > 0) { e.preventDefault(); setFocusOpen(true); return; }
        if (k === 'p') { e.preventDefault(); togglePin(state.part); return; }
        if (k === 'd') { e.preventDefault(); toggleDone(state.part); return; }
      }

      if (k === 'f') {
        const first = data?.flows.find((fl) => fl.view === state.view);
        if (first) { e.preventDefault(); setState((s) => ({ ...s, flow: first.id })); }
        return;
      }

      const dir = DIR_MAP[e.key];
      if (dir && activeView) {
        e.preventDefault();
        const next = navigate(activeView, state.part, dir);
        if (next) selectPart(next);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [
    data, state, activeView, cmdOpen, helpOpen, focusOpen, codeFiles,
    setState, switchView, selectPart, climb, drillInto, togglePin, toggleDone,
  ]);

  const getPartRect = useCallback((id: string) => {
    const node = diagramRef.current?.querySelector<HTMLElement>(`[data-part-id="${id}"]`);
    return node ? node.getBoundingClientRect() : null;
  }, []);

  const getCanvasRect = () => diagramRef.current?.getBoundingClientRect() ?? null;

  if (!data || !activeView) {
    return <div className="loading">Loading…</div>;
  }

  const pinnedParts = state.pinned.map((id) => data.partsById[id]).filter(Boolean);
  const showCompare = pinnedParts.length === 2;
  const activeFlow = state.flow ? data.flows.find((f) => f.id === state.flow) ?? null : null;

  return (
    <div className="app-shell">
      <Topbar
        views={data.views}
        activeView={state.view}
        arch={state.arch}
        legend={state.legend}
        progressDone={doneIds.size}
        progressTotal={data.parts.length}
        onArchChange={(arch) => setState((s) => ({ ...s, arch }))}
        onLegendChange={(legend) => setState((s) => ({ ...s, legend }))}
        onViewChange={switchView}
        onResetProgress={resetProgress}
        onOpenHelp={() => setHelpOpen(true)}
      />
      <main className="app-main">
        <section className="app-diagram" ref={diagramRef}>
          <Breadcrumb
            trail={state.trail}
            currentView={activeView}
            viewsById={data.viewsById}
            onClimb={climb}
          />
          <Diagram
            view={activeView}
            partsById={visiblePartsById}
            selectedId={state.part}
            highlightedIds={highlightedIds}
            doneIds={doneIds}
            legend={state.legend}
            moduleStatusHas={moduleStatusHas}
            fadedIds={fadedIds}
            onSelect={selectPart}
            onHover={setHoverId}
          />
          <FlowOverlay
            flow={activeFlow}
            onClose={() => setState((s) => ({ ...s, flow: null }))}
            getPartRect={getPartRect}
            canvasRect={getCanvasRect()}
          />
          <Legend mode={state.legend} />
        </section>
        {showCompare ? (
          <aside className="detail-panel">
            <div className="section-label">COMPARE</div>
            <PinCompare
              a={pinnedParts[0]}
              b={pinnedParts[1]}
              onUnpin={(id) => setState((s) => ({ ...s, pinned: s.pinned.filter((x) => x !== id) }))}
            />
          </aside>
        ) : (
          <DetailPanel
            data={data}
            part={selectedPart}
            done={!!selectedPart && doneIds.has(selectedPart.id)}
            onToggleDone={() => selectedPart && toggleDone(selectedPart.id)}
            onJump={selectPart}
            codeSection={
              <CodeTabs
                files={codeFiles}
                hoveredPartId={hoverId}
                onLineHover={setCodeHoverId}
                onExpand={codeFiles.length > 0 ? () => setFocusOpen(true) : undefined}
              />
            }
          />
        )}
      </main>
      <Tooltip
        part={hoverPart}
        legendClass={hoverPart ? colorClassFor(hoverPart, state.legend, moduleStatusHas(hoverPart.id)) : ''}
      />
      <CommandPalette
        data={data}
        open={cmdOpen}
        onClose={() => setCmdOpen(false)}
        onPickPart={selectPart}
        onPickView={switchView}
      />
      <HelpOverlay open={helpOpen} onClose={() => setHelpOpen(false)} />
      <CodeFocus
        open={focusOpen}
        part={selectedPart}
        files={codeFiles}
        hoveredPartId={hoverId}
        onLineHover={setCodeHoverId}
        onClose={() => setFocusOpen(false)}
      />
    </div>
  );
}
