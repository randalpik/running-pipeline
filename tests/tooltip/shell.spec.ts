// Regression net for the mobile shell: #rp-stage rotation, hamburger drawer,
// and the desktop no-op guarantee (src/plotting/_scaffold/shell.css/.js +
// build_shell.py).
//
// Drives the REAL built output/index.html (skips when absent). The mobile
// describe emulates a 390x664 portrait touch phone; taps are dispatched at
// SCREEN coordinates so the rotated-iframe hit-testing itself is under test —
// the single most important invariant of the shell design (verified by hand
// on Chromium during development; iOS Safari needs the scratch/mobile_spike
// device check).
//
// Rotation math (transform: rotate(90deg) translateY(-100%), origin 0 0, on
// a 390-wide portrait viewport): iframe-local (u, v) -> screen (390 - v, u),
// so a screen tap at (x, y) must arrive inside the iframe at
// clientX = y, clientY = 390 - x.

import { test, expect, Page } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const here = path.dirname(fileURLToPath(import.meta.url));
const SHELL = path.resolve(here, '../../output/index.html');

// Runs only under the chromium project (the mobile project's testMatch is
// touch.spec.ts alone); each describe sets its own viewport.

async function loadShell(page: Page) {
  await page.goto('file://' + SHELL);
  // Wait for the default tab's iframe to exist and load.
  await page.waitForSelector('#frame-wrap iframe.active');
  await page.waitForTimeout(400);
}

test.describe('desktop shell is unchanged', () => {
  test.skip(!fs.existsSync(SHELL),
    'output/index.html not built — run build_shell.py first');
  test.use({ viewport: { width: 1440, height: 900 } });

  test('stage is a no-op, tab bar visible, frame at y=36', async ({ page }) => {
    await loadShell(page);
    const r = await page.evaluate(() => {
      const stage = document.getElementById('rp-stage')!;
      const btn = document.getElementById('rp-menu-btn')!;
      const fw = document.getElementById('frame-wrap')!.getBoundingClientRect();
      const tabs = Array.from(document.querySelectorAll('#tabbar button.tab'))
        .map(b => getComputedStyle(b).display);
      return {
        stageDisplay: getComputedStyle(stage).display,
        btnDisplay: getComputedStyle(btn).display,
        frameTop: fw.top, frameW: fw.width,
        anyRegularTabHidden: tabs.slice(0, -1).some(d => d === 'none'),
      };
    });
    expect(r.stageDisplay).toBe('contents');
    expect(r.btnDisplay).toBe('none');
    expect(r.frameTop).toBe(36);
    expect(r.frameW).toBe(1440);
    expect(r.anyRegularTabHidden).toBe(false);
  });
});

test.describe('mobile portrait shell', () => {
  test.skip(!fs.existsSync(SHELL),
    'output/index.html not built — run build_shell.py first');
  test.use({
    viewport: { width: 390, height: 664 },
    hasTouch: true,
    isMobile: true,
  });

  test('stage rotates; iframe viewport is landscape; isVisible contract holds',
      async ({ page }) => {
    await loadShell(page);
    const r = await page.evaluate(() => {
      const stage = document.getElementById('rp-stage')!;
      const sr = stage.getBoundingClientRect();
      const bar = document.getElementById('tabbar')!;
      return {
        transform: getComputedStyle(stage).transform,
        stageRect: [Math.round(sr.left), Math.round(sr.top),
                    Math.round(sr.width), Math.round(sr.height)].join(','),
        barVisibility: getComputedStyle(bar).visibility,
        // shell.js isVisible() reads per-button computed display — the
        // closed drawer (visibility-hidden, transformed) must not zero it.
        // The admin-only tab is display:none by design until /api/auth/me.
        buttonDisplays: Array.from(
          bar.querySelectorAll('button.tab:not(.admin-only)'))
          .map(b => getComputedStyle(b).display),
      };
    });
    expect(r.transform).toBe('matrix(0, 1, -1, 0, 390, 0)');
    expect(r.stageRect).toBe('0,0,390,664');
    expect(r.barVisibility).toBe('hidden');
    for (const d of r.buttonDisplays) expect(d).not.toBe('none');

    const frame = page.frames().find(f => f !== page.mainFrame())!;
    const inner = await frame.evaluate(() => ({
      w: window.innerWidth, h: window.innerHeight,
      framed: document.documentElement.classList.contains('rp-framed'),
      mobile: document.documentElement.classList.contains('rp-mobile'),
    }));
    expect(inner.w).toBe(664);   // landscape while the device is portrait
    expect(inner.h).toBe(390);
    expect(inner.framed).toBe(true);
    expect(inner.mobile).toBe(true);   // pushed via rp-shell-mode
  });

  test('taps map through the rotation into iframe-local coordinates',
      async ({ page }) => {
    await loadShell(page);
    const frame = page.frames().find(f => f !== page.mainFrame())!;
    await frame.evaluate(() => {
      (window as any).__taps = [];
      document.addEventListener('touchstart', (e: any) => {
        const t = e.touches[0];
        (window as any).__taps.push([t.clientX, t.clientY]);
      }, true);
    });
    // Screen (100, 300) -> iframe-local (clientY=390-100=290... ) — mapping:
    // clientX = screenY, clientY = 390 - screenX.
    await page.touchscreen.tap(100, 300);
    await page.waitForTimeout(150);
    const taps = await frame.evaluate(() => (window as any).__taps);
    expect(taps.length).toBeGreaterThan(0);
    const [cx, cy] = taps[0];
    expect(Math.abs(cx - 300)).toBeLessThanOrEqual(1);
    expect(Math.abs(cy - (390 - 100))).toBeLessThanOrEqual(1);
  });

  test('position:fixed inside the iframe is unaffected by the parent transform',
      async ({ page }) => {
    await loadShell(page);
    const frame = page.frames().find(f => f !== page.mainFrame())!;
    const r = await frame.evaluate(() => {
      const el = document.createElement('div');
      el.style.cssText = 'position:fixed;right:0;bottom:0;width:10px;height:10px';
      document.body.appendChild(el);
      const b = el.getBoundingClientRect();
      el.remove();
      return { right: b.right, bottom: b.bottom,
               w: window.innerWidth, h: window.innerHeight };
    });
    expect(r.right).toBe(r.w);
    expect(r.bottom).toBe(r.h);
  });

  test('drawer: hamburger opens, tab tap switches + closes, scrim closes',
      async ({ page }) => {
    await loadShell(page);
    // Hamburger is at stage-local (0,0..44,44) -> screen top-right strip.
    await page.touchscreen.tap(390 - 22, 22);
    await page.waitForTimeout(300);
    expect(await page.evaluate(() =>
      document.body.classList.contains('rp-menu-open'))).toBe(true);

    // Tap a tab near the TOP of the list by its post-transform bounding box
    // (axis-aligned in portrait screen coords). Lower tabs sit past the
    // drawer's internal scroll fold on a 390px-tall stage, so their boxes
    // are clipped away — the drawer scrolls to reach them.
    const spot = await page.evaluate(() => {
      const b = document.querySelector(
        '#tabbar button.tab[data-slug="race_pace_all"]')!.getBoundingClientRect();
      return { x: b.left + b.width / 2, y: b.top + b.height / 2 };
    });
    await page.touchscreen.tap(spot.x, spot.y);
    await page.waitForTimeout(400);
    const afterTab = await page.evaluate(() => ({
      open: document.body.classList.contains('rp-menu-open'),
      active: (document.querySelector('#frame-wrap iframe.active') as any)
        ?.dataset.slug,
    }));
    expect(afterTab.open).toBe(false);
    expect(afterTab.active).toBe('race_pace_all');

    // Open again, then scrim tap (stage-local far corner = screen bottom-left
    // area, away from the drawer's 300px-wide strip).
    await page.touchscreen.tap(390 - 22, 22);
    await page.waitForTimeout(300);
    await page.touchscreen.tap(30, 620);
    await page.waitForTimeout(300);
    expect(await page.evaluate(() =>
      document.body.classList.contains('rp-menu-open'))).toBe(false);
  });
});
