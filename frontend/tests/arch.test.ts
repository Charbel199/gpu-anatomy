import { describe, it, expect } from 'vitest';
import { isVisibleInArch, ARCH_ORDER } from '../src/lib/arch';
import type { Part } from '../src/types';

const mk = (arch_introduced?: Part['arch_introduced']): Part => ({
  id: 'x', number: 0, name: 'x', category: 'compute', views: ['sm'],
  short: '', arch_introduced,
});

describe('arch filtering', () => {
  it('ARCH_ORDER starts at Fermi and ends at Blackwell', () => {
    expect(ARCH_ORDER[0]).toBe('Fermi');
    expect(ARCH_ORDER[ARCH_ORDER.length - 1]).toBe('Blackwell');
  });

  it('parts without arch_introduced are always visible', () => {
    expect(isVisibleInArch(mk(), 'Fermi')).toBe(true);
    expect(isVisibleInArch(mk(), 'Blackwell')).toBe(true);
  });

  it('Hopper-only parts hidden on Ampere', () => {
    expect(isVisibleInArch(mk('Hopper'), 'Ampere')).toBe(false);
  });

  it('Hopper-only parts visible on Hopper and later', () => {
    expect(isVisibleInArch(mk('Hopper'), 'Hopper')).toBe(true);
    expect(isVisibleInArch(mk('Hopper'), 'Blackwell')).toBe(true);
  });
});
