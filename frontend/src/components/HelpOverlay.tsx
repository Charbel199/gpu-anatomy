interface Props { open: boolean; onClose: () => void; }

const SHORTCUTS: Array<[string, string]> = [
  ['Ctrl+K  or  /', 'Open command palette'],
  ['arrows  or  h j k l', 'Move selection'],
  ['Enter', 'Drill in / focus code'],
  ['Esc', 'Drill out / clear selection / close overlays'],
  ['1 - 9', 'Switch view tab'],
  ['d', 'Mark selected part as done'],
  ['p', 'Pin selected part'],
  ['f', 'Play flow animation for current view'],
  ['e', 'Expand code into focus mode'],
  ['+ / -', 'Resize code (in focus mode)'],
  ['?', 'Show this help'],
];

export function HelpOverlay({ open, onClose }: Props) {
  if (!open) return null;
  return (
    <div className="help-backdrop" onClick={onClose}>
      <div className="help-panel" onClick={(e) => e.stopPropagation()}>
        <h3>Keyboard shortcuts</h3>
        <table className="help-table">
          <tbody>
            {SHORTCUTS.map(([k, v]) => (
              <tr key={k}><th><kbd>{k}</kbd></th><td>{v}</td></tr>
            ))}
          </tbody>
        </table>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
