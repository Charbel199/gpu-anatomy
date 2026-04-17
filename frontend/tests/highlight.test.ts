import { describe, it, expect } from 'vitest';
import { computeHighlightSet } from '../src/lib/highlight';
import type { Part } from '../src/types';

const mkPart = (id: string, related: string[] = []): Part => ({
  id, number: 0, name: id, category: 'memory', views: ['sm'], short: '', related,
});

describe('computeHighlightSet', () => {
  it('returns empty set when no part is selected', () => {
    const parts = { a: mkPart('a'), b: mkPart('b') };
    expect(computeHighlightSet(null, parts)).toEqual(new Set());
  });

  it('returns related parts of the selected part', () => {
    const parts = {
      a: mkPart('a', ['b', 'c']),
      b: mkPart('b'),
      c: mkPart('c'),
    };
    expect(computeHighlightSet('a', parts)).toEqual(new Set(['b', 'c']));
  });

  it('does not include the selected part itself', () => {
    const parts = { a: mkPart('a', ['b']), b: mkPart('b') };
    expect(computeHighlightSet('a', parts).has('a')).toBe(false);
  });

  it('silently skips unknown related ids', () => {
    const parts = { a: mkPart('a', ['ghost']) };
    expect(computeHighlightSet('a', parts)).toEqual(new Set());
  });
});
