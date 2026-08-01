import { defineConfig, devices } from '@playwright/test';

// Two deterministic projects on one engine:
//  - chromium: desktop — all specs except the touch suite.
//  - mobile: the landscape-phone envelope every plot page sees inside the
//    rotated shell (880x400, touch). Matches the (max-height: 520px) mobile
//    breakpoint regardless of whether the engine emulates pointer:coarse —
//    the tap handlers bind unconditionally by design.
export default defineConfig({
  testDir: '.',
  fullyParallel: true,
  reporter: 'list',
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
      testIgnore: ['touch.spec.ts'],
    },
    {
      name: 'mobile',
      use: {
        viewport: { width: 880, height: 400 },
        hasTouch: true,
        isMobile: true,
      },
      testMatch: ['touch.spec.ts'],
    },
  ],
});
