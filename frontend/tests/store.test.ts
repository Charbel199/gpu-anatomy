import { describe, it, expect, beforeEach } from 'vitest';
import { parseURL, toURL, defaultState, type AppState } from '../src/lib/store';

describe('store URL sync', () => {
  beforeEach(() => {
    window.history.replaceState({}, '', '/');
  });

  it('returns default state when URL has no params', () => {
    expect(parseURL(new URLSearchParams(''))).toEqual(defaultState);
  });

  it('parses a populated URL', () => {
    const p = new URLSearchParams('view=sm&part=warp_scheduler&arch=Hopper');
    expect(parseURL(p)).toEqual<AppState>({
      view: 'sm',
      part: 'warp_scheduler',
      arch: 'Hopper',
      pinned: [],
      flow: null,
      legend: 'category',
      trail: [],
    });
  });

  it('parses pinned as comma-separated list', () => {
    const p = new URLSearchParams('pinned=l2_cache,shared_memory_l1');
    expect(parseURL(p).pinned).toEqual(['l2_cache', 'shared_memory_l1']);
  });

  it('serializes back to a URL that round-trips', () => {
    const s: AppState = {
      view: 'sm',
      part: 'warp_scheduler',
      arch: 'Hopper',
      pinned: ['l2_cache'],
      flow: 'global_load',
      legend: 'category',
      trail: ['chip'],
    };
    const params = toURL(s);
    expect(parseURL(params)).toEqual(s);
  });

  it('omits default values from URL', () => {
    const params = toURL(defaultState);
    expect(params.toString()).toBe('');
  });
});
