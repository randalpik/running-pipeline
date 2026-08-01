// Tap-as-hover + mobile layout engine, on the REAL built pages, under the
// `mobile` project (880x400 touch — the landscape-phone envelope the rotated
// shell gives every plot iframe; see playwright.config.ts).
//
// Invariants:
//   1. A tap runs the same smooth/snap decision as hover: on a marker ->
//      snap; in open plot area -> smooth "nearest" tooltip, placed ABOVE the
//      finger; second tap same spot toggles off; tap in the margin hides.
//   2. Plotly's native touch interactions survive the tap layer: a touch
//      DRAG over the plot draws the zoom box and applies the zoom
//      (dragmode is never disabled), and a tap on a legend item toggles
//      its trace (tap_hover never preventDefaults, so the emulated click
//      reaches Plotly). Scrolling hides the fixed-position tooltip.
//   3. Reshaped pages (race_pace_by_distance transposed grid, recovery 2x1,
//      misc-trends taller stack) enter scroll mode with the figure at
//      --rp-plot-h, and traces keep their desktop axis assignments (only
//      domains move); margins/legends stay at their desktop spots.

import { test, expect, Page } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const here = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(here, '../../output');
const page_ = (name: string) => path.join(OUT, name);

async function ready(page: Page, file: string) {
  await page.goto('file://' + file);
  await page.waitForFunction(() => {
    const gd = document.querySelector('.plotly-graph-div') as any;
    return gd && gd._fullLayout && gd._fullLayout._size;
  });
  await page.waitForTimeout(600);   // mobile.js boot + newPlot settle
}

// A real touch drag. page.touchscreen has no drag primitive, and this is the
// only way to exercise the touch_scroll / Plotly split the way a finger does.
async function drag(page: Page, x0: number, y0: number, x1: number, y1: number) {
  const cdp = await page.context().newCDPSession(page);
  const n = 10;
  const pt = (i: number) => ([{
    x: x0 + (x1 - x0) * (i / n), y: y0 + (y1 - y0) * (i / n), id: 1 }]);
  await cdp.send('Input.dispatchTouchEvent' as any,
    { type: 'touchStart', touchPoints: pt(0) } as any);
  for (let i = 1; i <= n; i++) {
    await cdp.send('Input.dispatchTouchEvent' as any,
      { type: 'touchMove', touchPoints: pt(i) } as any);
    await page.waitForTimeout(20);
  }
  await cdp.send('Input.dispatchTouchEvent' as any,
    { type: 'touchEnd', touchPoints: [] } as any);
  await page.waitForTimeout(700);   // let any fling settle
  await cdp.detach();
}

function ttState(page: Page) {
  return page.evaluate(() => {
    const tt = document.querySelector('.rp-tooltip') as HTMLElement;
    const st = (window as any).__RP_TT_STATE || null;
    return {
      visible: tt && tt.style.display === 'block',
      box: tt ? tt.getBoundingClientRect() : null,
      isSnap: st && st.isSnap,
      place: st && st.place,
    };
  });
}

test.describe('tap acts as hover (fitness page)', () => {
  const FILE = page_('cs_timeline.html');
  test.skip(!fs.existsSync(FILE), 'cs_timeline.html not built');

  test('smooth tap: tooltip above the finger; toggle-off; margin hides; dragmode intact',
      async ({ page }) => {
    await ready(page, FILE);
    const spot = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      const rect = gd.getBoundingClientRect();
      const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
      return { x: rect.left + xa._offset + xa._length * 0.6,
               y: rect.top + ya._offset + ya._length * 0.55 };
    });
    await page.touchscreen.tap(spot.x, spot.y);
    await page.waitForTimeout(250);
    let s = await ttState(page);
    expect(s.visible).toBe(true);
    expect(s.place).toBe('touch');
    // Above the finger when it fits (box height + 24px gap <= tap y);
    // otherwise the placement legitimately flips below the finger.
    if (s.box!.height + 24 <= spot.y) {
      expect(s.box!.bottom).toBeLessThanOrEqual(spot.y);
    } else {
      expect(s.box!.top).toBeGreaterThanOrEqual(0);
    }
    // Touch must NOT disable Plotly zoom (drag-to-zoom is wanted on mobile).
    expect(await page.evaluate(() =>
      (document.querySelector('.plotly-graph-div') as any)._fullLayout.dragmode
    )).toBe('zoom');

    await page.touchscreen.tap(spot.x, spot.y);   // same spot -> toggle off
    await page.waitForTimeout(250);
    s = await ttState(page);
    expect(s.visible).toBe(false);

    await page.touchscreen.tap(spot.x, spot.y + 8);   // fresh tap, re-shows
    await page.waitForTimeout(250);
    expect((await ttState(page)).visible).toBe(true);

    await page.touchscreen.tap(spot.x, 4);   // far margin (above plot) hides
    await page.waitForTimeout(250);
    expect((await ttState(page)).visible).toBe(false);
  });

  test('tap on a snap-eligible marker snaps', async ({ page }) => {
    await ready(page, FILE);
    const marker = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      const fl = gd._fullLayout;
      const rect = gd.getBoundingClientRect();
      const t = gd.data.find((tr: any) =>
        tr.meta && tr.meta.snap_eligible === true && tr.visible !== false);
      if (!t) return null;
      const xs = (t.x as any)._inputArray || t.x;
      const ys = (t.y as any)._inputArray || t.y;
      const i = Math.floor(xs.length / 2);
      const xa = fl.xaxis, ya = fl.yaxis;
      const xv = typeof xs[i] === 'string' ? new Date(xs[i]).getTime() : xs[i];
      return { x: rect.left + xa._offset + xa.c2p(xv),
               y: rect.top + ya._offset + ya.c2p(ys[i]) };
    });
    test.skip(!marker, 'no snap-eligible trace with data');
    await page.touchscreen.tap(marker!.x, marker!.y);
    await page.waitForTimeout(250);
    const s = await ttState(page);
    expect(s.visible).toBe(true);
    expect(s.isSnap).toBe(true);
  });

  test('touch drag draws Plotly zoom box and applies the zoom', async ({ page }) => {
    await ready(page, FILE);
    const cdp = await page.context().newCDPSession(page);
    const geom = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      const r = gd.getBoundingClientRect();
      const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
      return {
        x0: r.left + xa._offset + xa._length * 0.3,
        x1: r.left + xa._offset + xa._length * 0.7,
        y: r.top + ya._offset + ya._length * 0.5,
        range: gd._fullLayout.xaxis.range.join('|'),
      };
    });
    const touch = (type: string, pts: { x: number, y: number }[]) =>
      cdp.send('Input.dispatchTouchEvent', {
        type, touchPoints: pts.map(p => ({ x: p.x, y: p.y, id: 1 })),
      } as any);
    await touch('touchStart', [{ x: geom.x0, y: geom.y }]);
    for (let i = 1; i <= 10; i++) {
      await touch('touchMove',
        [{ x: geom.x0 + (geom.x1 - geom.x0) * (i / 10), y: geom.y }]);
      await page.waitForTimeout(20);
    }
    const midbox = await page.evaluate(() =>
      !!document.querySelector('.plotly-graph-div .zoomlayer .zoombox'));
    await touch('touchEnd', []);
    await page.waitForTimeout(500);
    const after = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      return { range: gd._fullLayout.xaxis.range.join('|'),
               tt: (document.querySelector('.rp-tooltip') as HTMLElement)
                 .style.display };
    });
    expect(midbox).toBe(true);
    expect(after.range).not.toBe(geom.range);
    expect(after.tt).not.toBe('block');   // drag hides the tooltip
  });

  test('tap on a legend item toggles its trace', async ({ page }) => {
    await ready(page, FILE);
    const leg = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      const item = gd.querySelector('.infolayer .legend .traces');
      if (!item) return null;
      const b = item.getBoundingClientRect();
      return { x: b.left + b.width / 2, y: b.top + b.height / 2 };
    });
    test.skip(!leg, 'no legend on this page');
    const visibles = () => page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      return gd.data.map((t: any) => String(t.visible)).join(',');
    });
    const before = await visibles();
    await page.touchscreen.tap(leg!.x, leg!.y);
    await page.waitForTimeout(600);
    const after = await visibles();
    expect(after).not.toBe(before);
  });
});

test.describe('race distances mobile reshape', () => {
  const FILE = page_('race_pace_by_distance.html');
  test.skip(!fs.existsSync(FILE), 'race_pace_by_distance.html not built');

  test('scroll mode + transposed grid + scroll hides tooltip', async ({ page }) => {
    await ready(page, FILE);
    const r = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      const cssH = getComputedStyle(document.documentElement)
        .getPropertyValue('--rp-plot-h').trim();
      return {
        scroll: document.body.classList.contains('rp-scroll'),
        pending: document.documentElement.classList.contains('rp-mobile-pending'),
        clientH: gd.clientHeight, cssH,
        scrollable: document.documentElement.scrollHeight > window.innerHeight,
        // Transposed grid: xaxis3 (cell 3) shares column 1's x-domain; in the
        // desktop 2x4 grid it's row1col3 with a distinct domain.
        sameCol: Math.abs(gd.layout.xaxis3.domain[0] - gd.layout.xaxis.domain[0]) < 1e-9,
        // The legend keeps its desktop spot (vertical, right rail).
        legendOrient: (gd.layout.legend || {}).orientation,
        // Trace->axis assignments must be the desktop ones (reshape is
        // layout-only).
        axes: gd.data.map((t: any) => (t.xaxis || 'x')).join(','),
      };
    });
    expect(r.scroll).toBe(true);
    expect(r.pending).toBe(false);
    expect(r.clientH).toBe(parseInt(r.cssH, 10));
    expect(r.scrollable).toBe(true);
    expect(r.sameCol).toBe(true);
    expect(r.legendOrient).not.toBe('h');
    expect(r.axes.length).toBeGreaterThan(0);

    // Tap -> tooltip; scroll -> hidden (fixed-position tooltip goes stale).
    const spot = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      const rect = gd.getBoundingClientRect();
      const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
      return { x: rect.left + xa._offset + xa._length / 2,
               y: rect.top + ya._offset + ya._length / 2 };
    });
    await page.touchscreen.tap(spot.x, spot.y);
    await page.waitForTimeout(250);
    expect((await ttState(page)).visible).toBe(true);
    await page.evaluate(() => window.scrollTo(0, 300));
    await page.waitForTimeout(250);
    expect((await ttState(page)).visible).toBe(false);
  });

  // The production bug: these pages were unreachable below the fold. Neither
  // the browser nor Plotly scrolls them (Plotly claims plot-area drags for
  // its zoom box; Chromium won't pan anything under the shell's rotated
  // stage), so _scaffold/touch_scroll.js does it — for drags that start off
  // Plotly's draglayer, leaving panel interiors to Plotly.
  test('margin drag scrolls the page; panel drag stays Plotly zoom',
      async ({ page }) => {
    await ready(page, FILE);
    const geom = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      const r = gd.getBoundingClientRect();
      const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
      return {
        marginX: r.left + gd._fullLayout.margin.l / 2,   // left of the y axis
        panelX: r.left + xa._offset + xa._length * 0.5,
        yTop: r.top + ya._offset + ya._length * 0.25,
        yBot: r.top + ya._offset + ya._length * 0.75,
        yr: gd._fullLayout.yaxis.range.join('|'),
      };
    });

    await drag(page, geom.marginX, geom.yBot, geom.marginX, geom.yTop);
    const afterMargin = await page.evaluate(() => ({
      y: Math.round(window.scrollY),
      yr: (document.querySelector('.plotly-graph-div') as any)
        ._fullLayout.yaxis.range.join('|'),
      dragging: !!(document.querySelector('.plotly-graph-div') as any)._dragging,
    }));
    expect(afterMargin.y).toBeGreaterThan(100);      // scrolled
    expect(afterMargin.yr).toBe(geom.yr);            // did not zoom
    expect(afterMargin.dragging).toBe(false);        // plotly not wedged

    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(250);
    const before = await page.evaluate(() => ({
      y: Math.round(window.scrollY),
      yr: (document.querySelector('.plotly-graph-div') as any)
        ._fullLayout.yaxis.range.join('|'),
    }));
    await drag(page, geom.panelX, geom.yBot, geom.panelX, geom.yTop);
    const afterPanel = await page.evaluate(() => ({
      y: Math.round(window.scrollY),
      yr: (document.querySelector('.plotly-graph-div') as any)
        ._fullLayout.yaxis.range.join('|'),
      dragging: !!(document.querySelector('.plotly-graph-div') as any)._dragging,
    }));
    expect(afterPanel.yr).not.toBe(before.yr);       // zoomed
    expect(afterPanel.y).toBe(before.y);             // did not scroll
    expect(afterPanel.dragging).toBe(false);
  });
});

test.describe('recovery mobile reshape', () => {
  const FILE = page_('recovery_pace.html');
  test.skip(!fs.existsSync(FILE), 'recovery_pace.html not built');

  test('panels stack; norm-filter keeps its fixed right-rail spot', async ({ page }) => {
    await ready(page, FILE);
    const r = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      const nf = document.getElementById('norm-filter')!;
      const b = nf.getBoundingClientRect();
      return {
        scroll: document.body.classList.contains('rp-scroll'),
        stacked: gd.layout.yaxis.domain[0] > gd.layout.yaxis2.domain[1],
        nfPosition: getComputedStyle(nf).position,
        nfRight: b.right >= window.innerWidth - 20,
        nfTop: b.top,
      };
    });
    expect(r.scroll).toBe(true);
    expect(r.stacked).toBe(true);
    expect(r.nfPosition).toBe('fixed');   // right rail, same as desktop
    expect(r.nfRight).toBe(true);
    expect(r.nfTop).toBe(64);   // dropped below the subtitle line on mobile
  });
});

test.describe('misc trends mobile heights', () => {
  const FILE = page_('qualitative_trends.html');
  test.skip(!fs.existsSync(FILE), 'qualitative_trends.html not built');

  test('taller stack, page toggle keeps patch, insets track, gridlines live',
      async ({ page }) => {
    await ready(page, FILE);
    expect(await page.evaluate(() =>
      document.body.classList.contains('rp-scroll'))).toBe(true);
    const h1 = await page.evaluate(() =>
      (document.querySelector('.plotly-graph-div') as any).clientHeight);
    expect(h1).toBeGreaterThan(500);

    await page.evaluate(() => {
      (document.querySelector(
        '#trends-toggle .rp-btn-pill[data-value="other"]') as HTMLElement).click();
    });
    await page.waitForTimeout(700);
    const r = await page.evaluate(() => {
      const gd = document.querySelector('.plotly-graph-div') as any;
      const gdR = gd.getBoundingClientRect();
      const insets = Array.from(document.querySelectorAll('.rp-inset'))
        .filter(el => (el as HTMLElement).style.display !== 'none');
      return {
        h: gd.clientHeight,
        scroll: document.body.classList.contains('rp-scroll'),
        insetsInside: insets.every(el => {
          const b = el.getBoundingClientRect();
          return b.top >= gdR.top - 1 && b.bottom <= gdR.bottom + 1;
        }),
        gridPaths: document.querySelectorAll(
          '.plotly-graph-div .gridlayer path').length,
      };
    });
    expect(r.h).toBeGreaterThan(500);
    expect(r.scroll).toBe(true);
    expect(r.insetsInside).toBe(true);
    expect(r.gridPaths).toBeGreaterThan(10);   // the plotly gridline scar
  });
});
