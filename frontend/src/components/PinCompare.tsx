import type { Part } from '../types';

interface Props {
  a: Part;
  b: Part;
  onUnpin: (id: string) => void;
}

function countLabel(p: Part): string {
  return p.count ? `${p.count.n} per ${p.count.per}` : '-';
}

export function PinCompare({ a, b, onUnpin }: Props) {
  const rows: Array<[string, string, string]> = [
    ['#', String(a.number), String(b.number)],
    ['Category', a.category, b.category],
    ['Count', countLabel(a), countLabel(b)],
    ['Since', a.arch_introduced ?? '-', b.arch_introduced ?? '-'],
    ['Summary', a.short, b.short],
  ];

  return (
    <div className="pincompare">
      <div className="pin-head">
        <div>
          <span className={`chip chip-readonly cat-${a.category}`}>{a.name}</span>
          <button onClick={() => onUnpin(a.id)} className="pin-x" aria-label={`Unpin ${a.name}`}>×</button>
        </div>
        <div>
          <span className={`chip chip-readonly cat-${b.category}`}>{b.name}</span>
          <button onClick={() => onUnpin(b.id)} className="pin-x" aria-label={`Unpin ${b.name}`}>×</button>
        </div>
      </div>
      <table className="pin-table">
        <tbody>
          {rows.map(([label, av, bv]) => (
            <tr key={label}>
              <th>{label}</th>
              <td>{av}</td>
              <td>{bv}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
