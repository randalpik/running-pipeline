// Regression net for the Misc Trends drag-to-zoom
// (src/plots/plot_qualitative_trends.js + the __rpZoomDragging guard in
// src/plotting/_scaffold/cursor_tooltip.js).
//
// Unlike tooltip.spec.ts (blank-page scaffold fixtures), this drives the REAL
// built page — a drag is meaningless without live Plotly axes — so it needs
// `python src/plots/plot_qualitative_trends.py` to have run first. The suite
// skips (not fails) when output/qualitative_trends.html is absent.
//
// Invariants:
//   1. window.__rpZoomDragging suppresses the tooltip (scaffold guard), and
//      a plain hover still shows it (guard is a no-op when unset).
//   2. A synthetic x-drag shrinks the master x-range, moves every matched
//      axis with it, and reveals the reset pill.
//   3. Double-click restores the home range and hides the pill.
//   4. Zooming swaps every envelope image for a fresh client-side canvas
//      render covering exactly the zoomed window (source/x/sizex change);
//      reset restores the baked full-range PNGs byte-identically.

import { test, expect, Page } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGE = path.resolve(here, '../../output/qualitative_trends.html');

test.describe('misc-trends drag zoom', () => {
  test.skip(!fs.existsSync(PAGE),
    'output/qualitative_trends.html not built — run plot_qualitative_trends.py first');

  async function ready(page: Page) {
    await page.goto('file://' + PAGE);
    await page.waitForFunction(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      return gd && gd._fullLayout && gd._fullLayout._size;
    });
  }

  // Live geometry + zoom state, read fresh each call.
  function state(page: Page) {
    return page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      const fl = gd._fullLayout;
      const ks = Object.keys(fl).filter(k => /^xaxis\d*$/.test(k));
      const master = ks.find(k => !fl[k].matches) || 'xaxis';
      const rect = gd.getBoundingClientRect();
      const sz = fl._size;
      const reset = document.querySelector('.rp-zoom-reset');
      const tt = document.querySelector('.rp-tooltip');
      return {
        range: fl[master].range.join('|'),
        allEqual: ks.every(k => fl[k].range.join('|') === fl[master].range.join('|')),
        resetVisible: reset ? getComputedStyle(reset).display !== 'none' : null,
        ttVisible: tt ? getComputedStyle(tt).display !== 'none' : null,
        area: { left: rect.left + sz.l, top: rect.top + sz.t, w: sz.w, h: sz.h },
      };
    });
  }

  // Baked-vs-live image state: data-URI lengths stand in for content
  // (byte-equal restore ⇒ equal length; a re-render at a different window
  // and resolution virtually never collides).
  function images(page: Page) {
    return page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      return ((gd.layout && gd.layout.images) || []).map((im: any) => ({
        len: String(im.source || '').length, sizex: im.sizex,
      }));
    });
  }

  test('drag zooms + re-rasters, pill shows, dblclick resets all', async ({ page }) => {
    await ready(page);
    const s0 = await state(page);
    const baked = await images(page);
    expect(s0.resetVisible).toBe(false);
    expect(baked.length).toBeGreaterThan(0);

    // Hover y must land inside a subplot row, not the inter-panel gap —
    // 0.42 of the plot height is inside row 2 on the 4-row weather page.
    const cy = s0.area.top + s0.area.h * 0.42;
    await page.mouse.move(s0.area.left + s0.area.w * 0.45, cy);
    await page.mouse.down();
    await page.mouse.move(s0.area.left + s0.area.w * 0.55, cy, { steps: 5 });
    await page.mouse.up();
    await page.waitForTimeout(400);

    const zoomed = await state(page);
    expect(zoomed.range).not.toBe(s0.range);
    expect(zoomed.allEqual).toBe(true);
    expect(zoomed.resetVisible).toBe(true);
    // Every envelope raster was re-rendered client-side for the new window.
    const live = await images(page);
    for (let k = 0; k < baked.length; k++) {
      expect(live[k].len).not.toBe(baked[k].len);
      expect(live[k].sizex).not.toBe(baked[k].sizex);
    }

    await page.mouse.dblclick(s0.area.left + s0.area.w * 0.5, cy);
    await page.waitForTimeout(400);
    const reset = await state(page);
    expect(reset.range).toBe(s0.range);
    expect(reset.resetVisible).toBe(false);
    // Baked full-range PNGs restored exactly.
    const restored = await images(page);
    for (let k = 0; k < baked.length; k++) {
      expect(restored[k].len).toBe(baked[k].len);
      expect(restored[k].sizex).toBe(baked[k].sizex);
    }
  });

  test('__rpZoomDragging suppresses the tooltip; plain hover shows it', async ({ page }) => {
    await ready(page);
    const s0 = await state(page);
    const cx = s0.area.left + s0.area.w * 0.8;  // 2024ish — has data
    const cy = s0.area.top + s0.area.h * 0.42;

    await page.mouse.move(cx, cy);
    await page.mouse.move(cx + 6, cy);
    await page.waitForTimeout(200);
    expect((await state(page)).ttVisible).toBe(true);

    await page.evaluate(() => { (window as any).__rpZoomDragging = true; });
    await page.mouse.move(cx + 12, cy);
    await page.waitForTimeout(200);
    expect((await state(page)).ttVisible).toBe(false);

    await page.evaluate(() => { (window as any).__rpZoomDragging = false; });
    await page.mouse.move(cx + 18, cy);
    await page.waitForTimeout(200);
    expect((await state(page)).ttVisible).toBe(true);
  });
});
