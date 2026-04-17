import { ARCH_ORDER } from '../lib/arch';
import type { Architecture } from '../types';

interface Props {
  value: Architecture;
  onChange: (a: Architecture) => void;
}

export function ArchSelect({ value, onChange }: Props) {
  return (
    <select
      className="arch-select"
      value={value}
      onChange={(e) => onChange(e.target.value as Architecture)}
      title="Architecture generation"
    >
      {ARCH_ORDER.map((a) => <option key={a} value={a}>{a}</option>)}
    </select>
  );
}
