import { defineConfig, devices } from '@playwright/test';

// Single deterministic browser; the tooltip is pure HTML/CSS/JS so one
// engine with a real layout pass is enough.
export default defineConfig({
  testDir: '.',
  fullyParallel: true,
  reporter: 'list',
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
