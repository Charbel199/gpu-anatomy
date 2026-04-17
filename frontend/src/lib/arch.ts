import type { Architecture, Part } from '../types';

export const ARCH_ORDER: Architecture[] = [
  'Fermi', 'Kepler', 'Maxwell', 'Pascal', 'Volta', 'Turing', 'Ampere', 'Hopper', 'Blackwell',
];

export function archIndex(a: Architecture): number {
  return ARCH_ORDER.indexOf(a);
}

export function isVisibleInArch(part: Part, selected: Architecture): boolean {
  if (!part.arch_introduced) return true;
  return archIndex(part.arch_introduced) <= archIndex(selected);
}
