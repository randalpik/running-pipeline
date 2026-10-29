// Shoes tab: bars, hit-test and whole-pair outline on the REAL built page.
//
// The page redraws every bar from its segment dates after each Plotly render
// (1px minimum width) and replaces the scaffold's point snapping with a
// rectangle hit-test (window.__TT_HIT_TEST). The first version indexed
// segments by the bar datum's `.p` — the bar's ROW on horizontal bars — which
// drew every segment of a pair on top of one segment. Ground truth here is the
// page's own segment list (window.__PLOT_SHOES) checked against Plotly's datum
// extents (s0/s1), independent of the redraw under test.
//
// Needs `python src/plots/plot_shoes.py` first; skips when the page is absent
// (a profile without shoe data never writes it).

import { test, expect, Page } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGE = path.resolve(here, '../../output/shoes.html');
const BAR_RE = String.raw`^M([-\d.]+),([-\d.]+)V([-\d.]+)H([-\d.]+)V([-\d.]+)Z$`;

test.skip(!fs.existsSync(PAGE), 'output/shoes.html not built');

// The live tooltip is the FIRST .rp-tooltip; the scaffold appends a hidden
// measuring clone with the same class.
const tooltip = (page: Page) => page.locator('.rp-tooltip').first();

async function ready(page: Page) {
  await page.setViewportSize({ width: 1400, height: 800 });
  await page.goto('file://' + PAGE);
  await page.waitForFunction(() => {
    const gd = document.querySelector('.plotly-graph-div') as any;
    return gd && gd._fullLayout && gd._fullLayout._size;
  });
  await page.waitForTimeout(900);
}

// Every drawn bar: its pair, segment index, pixel extent, and whether its
// geometry matches the segment Plotly bound to it.
function bars(page: Page) {
  return page.evaluate((reSrc) => {
    const S = (window as any).__PLOT_SHOES;
    const gd = document.querySelector('.plotly-graph-div') as any;
    const xa = gd._fullLayout.xaxis;
    const re = new RegExp(reSrc);
    const out: any[] = [];
    gd.querySelectorAll('.barlayer .trace').forEach((tr: any) => {
      const pair = tr.__data__[0].trace.index;
      tr.querySelectorAll('.point path').forEach((p: any) => {
        const m = (p.getAttribute('d') || '').match(re);
        const d = p.__data__;
        const seg = S.pairs[pair].segs[d.i];
        out.push({
          pair, i: d.i,
          ok: !!m,
          w: m ? parseFloat(m[4]) - parseFloat(m[1]) : NaN,
          left: m ? parseFloat(m[1]) : NaN,
          right: m ? parseFloat(m[4]) : NaN,
          // Plotly's own extent for this point vs the segment list.
          match: seg && seg[0] === d.s0 && seg[1] === d.s1,
          px0: xa.c2p(seg[0]), px1: xa.c2p(seg[1]),
          lit: p.style.strokeWidth === '1.5px',
        });
      });
    });
    return out;
  }, BAR_RE);
}

// Client point at the middle of segment k of the named pair.
function segPoint(page: Page, name: string, k: number) {
  return page.evaluate(([nm, kk]) => {
    const S = (window as any).__PLOT_SHOES;
    const gd = document.querySelector('.plotly-graph-div') as any;
    const fl = gd._fullLayout, r = gd.getBoundingClientRect();
    const i = S.pairs.findIndex((p: any) => p.html.includes(`>${nm}</div>`));
    const sg = S.pairs[i].segs[kk as number];
    return {
      i,
      x: r.left + fl.xaxis._offset + fl.xaxis.c2p((sg[0] + sg[1]) / 2),
      y: r.top + fl.yaxis._offset + fl.yaxis.c2p(S.pairs[i].row),
    };
  }, [name, k] as const);
}

// Client point at (date, row offset from the named pair's row).
function dataPoint(page: Page, name: string, date: string, dy = 0) {
  return page.evaluate(([nm, d, off]) => {
    const S = (window as any).__PLOT_SHOES;
    const gd = document.querySelector('.plotly-graph-div') as any;
    const fl = gd._fullLayout, r = gd.getBoundingClientRect();
    const p = S.pairs.find((q: any) => q.html.includes(`>${nm}</div>`));
    return {
      x: r.left + fl.xaxis._offset + fl.xaxis.c2p(Date.parse(d as string)),
      y: r.top + fl.yaxis._offset + fl.yaxis.c2p(p.row + (off as number)),
    };
  }, [name, date, dy] as const);
}

test('every segment is drawn once, at its own dates, at least 1px wide', async ({ page }) => {
  await ready(page);
  const b = await bars(page);
  const nSegs = await page.evaluate(() =>
    (window as any).__PLOT_SHOES.pairs.reduce((a: number, p: any) => a + p.segs.length, 0));
  expect(b.length).toBe(nSegs);
  expect(b.every((x) => x.ok && x.match)).toBe(true);
  expect(Math.min(...b.map((x) => x.w))).toBeGreaterThanOrEqual(0.999);
  // Placed at its own segment: left edge at the segment start unless the
  // 1px floor widened it (then centred on the segment).
  for (const x of b) {
    const centre = (x.px0 + x.px1) / 2;
    expect(Math.abs((x.left + x.right) / 2 - centre)).toBeLessThan(0.51);
  }
});

test('hover outlines every segment of the pair and only that pair', async ({ page }) => {
  await ready(page);
  const pt = await segPoint(page, 'Nike ZoomX Dragonfly', 5);
  await page.mouse.move(pt.x, pt.y);
  await page.waitForTimeout(300);
  const tt = tooltip(page);
  await expect(tt).toBeVisible();
  await expect(tt).toContainText('Nike ZoomX Dragonfly');
  await expect(tt).toContainText('PRs set');
  const b = await bars(page);
  const mine = b.filter((x) => x.pair === pt.i);
  expect(mine.length).toBeGreaterThan(1);
  expect(mine.every((x) => x.lit)).toBe(true);
  expect(b.filter((x) => x.pair !== pt.i).some((x) => x.lit)).toBe(false);

  // Same row, between pairs (the Endorphin 2 ended 2021-06, the Dragonfly
  // starts 2022-08): no region there, so the tooltip hides and the outline
  // clears.
  const empty = await dataPoint(page, 'Nike ZoomX Dragonfly', '2022-01-01');
  await page.mouse.move(empty.x, empty.y);
  await page.waitForTimeout(300);
  await expect(tt).toBeHidden();
  expect((await bars(page)).some((x) => x.lit)).toBe(false);
});

test("a pair's whole region is its hover target — no flashing across gaps", async ({ page }) => {
  await ready(page);
  // Sweep the Dragonfly's span (sparse: weeks between wears) at its row
  // centre and near both edges of the row band.
  const pts = await page.evaluate(() => {
    const S = (window as any).__PLOT_SHOES;
    const gd = document.querySelector('.plotly-graph-div') as any;
    const fl = gd._fullLayout, r = gd.getBoundingClientRect();
    const p = S.pairs.find((q: any) => q.html.includes('>Nike ZoomX Dragonfly</div>'));
    const s = p.segs[0][0], e = p.segs[p.segs.length - 1][1];
    const out: { x: number; y: number }[] = [];
    for (let k = 0; k <= 40; k++) {
      for (const dy of [-0.45, 0, 0.45]) {
        out.push({ x: r.left + fl.xaxis._offset + fl.xaxis.c2p(s + (e - s) * k / 40),
                   y: r.top + fl.yaxis._offset + fl.yaxis.c2p(p.row + dy) });
      }
    }
    return out;
  });
  for (const pt of pts) {
    await page.mouse.move(pt.x, pt.y);
    await page.waitForTimeout(40);
    const txt = await page.evaluate(() => {
      const st = (window as any).__RP_TT_STATE;
      return st && st.shown ? (document.querySelector('.rp-tooltip') as HTMLElement).innerText : '';
    });
    expect(txt).toContain('Nike ZoomX Dragonfly');
  }
});

test('zoomed in, bars track the axis and stay hoverable', async ({ page }) => {
  await ready(page);
  await page.evaluate(() => {
    const gd = document.querySelector('.plotly-graph-div') as any;
    return (window as any).Plotly.relayout(gd, { 'xaxis.range': ['2023-06-01', '2023-08-15'] });
  });
  await page.waitForTimeout(600);
  const b = await bars(page);
  expect(b.every((x) => x.ok && x.match)).toBe(true);
  expect(Math.min(...b.map((x) => x.w))).toBeGreaterThanOrEqual(0.999);
  // A single summer-2023 race day is now many pixels wide.
  const pt = await segPoint(page, 'Nike ZoomX Dragonfly', 16);
  await page.mouse.move(pt.x, pt.y);
  await page.waitForTimeout(300);
  await expect(tooltip(page)).toContainText('Nike ZoomX Dragonfly');
});

test('a brand hidden from the legend is not hoverable', async ({ page }) => {
  await ready(page);
  const pt = await segPoint(page, 'Nike ZoomX Dragonfly', 5);
  await page.locator('.legend .traces', { hasText: 'Nike' }).click();
  await page.waitForTimeout(500);
  await page.mouse.move(pt.x, pt.y);
  await page.waitForTimeout(300);
  await expect(tooltip(page)).toBeHidden();
});
