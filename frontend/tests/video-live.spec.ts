import path from 'node:path';

import { expect, test } from '@playwright/test';

test.skip(!process.env.LIVE_VIDEO_E2E, 'Set LIVE_VIDEO_E2E=1 with the local backends running.');

test('live video upload reaches the backend and returns a printable STL', async ({ page }) => {
  test.setTimeout(120_000);
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  const backendUrl = process.env.LIVE_BACKEND_URL || 'http://127.0.0.1:8004';
  const videoBackendUrl = process.env.LIVE_VIDEO_BACKEND_URL || 'http://127.0.0.1:8016';

  await page.goto(
    `/?backendUrl=${encodeURIComponent(backendUrl)}&videoBackendUrl=${encodeURIComponent(videoBackendUrl)}`,
  );
  await expect(page.getByText('Backend Runtime')).toBeVisible();
  await page.locator('input[type="file"]').setInputFiles(path.resolve('../output/video-live-smoke.avi'));
  await expect(page.getByRole('button', { name: 'Video' })).toHaveClass(/border-blue-700/);
  await expect(page.getByLabel('Selection model')).toHaveValue('turntable-grabcut');

  await page.getByRole('button', { name: 'Run' }).click();

  await expect(page.getByText('Printable video STL ready')).toBeVisible({ timeout: 90_000 });
  await expect(page.getByRole('link', { name: 'Download STL' })).toHaveAttribute('href', /output_model\.stl$/);
  await expect(page.getByText('Diagnostics', { exact: true })).toBeVisible();
  expect(pageErrors).toEqual([]);
});
