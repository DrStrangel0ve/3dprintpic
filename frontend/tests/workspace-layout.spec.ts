import { expect, type Page, test } from '@playwright/test';

async function mockBackendServices(page: Page) {
  await page.route('**/health', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        runtime: { device: 'Test GPU', torch: 'test', cuda_available: true, cuda_version: 'test' },
      }),
    }),
  );
  await page.route('**/depth/preload/depthpro/status', (route) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ status: 'missing' }) }),
  );
  await page.route('**/models', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        service: 'contract-drift-smoke',
        defaults: { image_to_mesh: 42 },
        groups: { selection: 'unexpected-shape', image_to_mesh: [null, { label: 'Missing id' }] },
        metrics: { watertightness: true },
      }),
    }),
  );
  await page.route('**/providers/image-to-mesh', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ providers: [null, { id: 'triposg', runnable: true, setup_errors: 'none' }] }),
    }),
  );
}

function capturePageErrors(page: Page) {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

const wideDesktopViewports = [
  { width: 1280, height: 720 },
  { width: 1440, height: 900 },
];

for (const viewport of wideDesktopViewports) {
  test(`desktop columns scroll without clipping at ${viewport.width}px`, async ({ page }) => {
    const pageErrors = capturePageErrors(page);
    await mockBackendServices(page);
    await page.setViewportSize(viewport);
    await page.goto('/');

    await expect(page.getByRole('heading', { name: 'Photo/Video to STL' })).toBeVisible();
    await expect(page.getByText('Model service')).toBeVisible();

    const metrics = await page.evaluate(() => ({
      document: {
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
        clientHeight: document.documentElement.clientHeight,
        scrollHeight: document.documentElement.scrollHeight,
      },
      columns: Array.from(document.querySelectorAll<HTMLElement>('[data-workspace-column]')).map((column) => ({
        name: column.dataset.workspaceColumn,
        clientWidth: column.clientWidth,
        scrollWidth: column.scrollWidth,
        clientHeight: column.clientHeight,
        scrollHeight: column.scrollHeight,
        overflowY: getComputedStyle(column).overflowY,
      })),
    }));

    expect(metrics.document.scrollWidth).toBeLessThanOrEqual(metrics.document.clientWidth);
    expect(metrics.document.scrollHeight).toBeLessThanOrEqual(metrics.document.clientHeight);
    expect(metrics.columns).toHaveLength(3);
    for (const column of metrics.columns) {
      expect(column.clientWidth).toBeGreaterThan(0);
      expect(column.scrollWidth).toBeLessThanOrEqual(column.clientWidth);
      expect(column.overflowY).toBe('auto');
    }

    const outputColumn = page.getByTestId('workspace-output');
    await outputColumn.evaluate((column) => {
      column.scrollTop = column.scrollHeight;
    });
    await expect(page.getByRole('heading', { name: 'Promotion Gates' })).toBeVisible();
    expect(await outputColumn.evaluate((column) => column.scrollTop)).toBeGreaterThan(0);
    expect(pageErrors).toEqual([]);
  });
}

test('laptop layout uses two readable columns without horizontal overflow', async ({ page }) => {
  const pageErrors = capturePageErrors(page);
  await mockBackendServices(page);
  await page.setViewportSize({ width: 1024, height: 720 });
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'Photo/Video to STL' })).toBeVisible();
  const metrics = await page.evaluate(() => ({
    document: {
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
      clientHeight: document.documentElement.clientHeight,
      scrollHeight: document.documentElement.scrollHeight,
    },
    columns: Object.fromEntries(
      Array.from(document.querySelectorAll<HTMLElement>('[data-workspace-column]')).map((column) => {
        const rect = column.getBoundingClientRect();
        return [
          column.dataset.workspaceColumn,
          {
            left: rect.left,
            top: rect.top,
            width: rect.width,
            overflowY: getComputedStyle(column).overflowY,
          },
        ];
      }),
    ),
  }));

  expect(metrics.document.scrollWidth).toBeLessThanOrEqual(metrics.document.clientWidth);
  expect(metrics.document.scrollHeight).toBeGreaterThan(metrics.document.clientHeight);
  expect(metrics.columns.input.left).toBe(metrics.columns.output.left);
  expect(metrics.columns.output.top).toBeGreaterThan(metrics.columns.input.top);
  expect(metrics.columns.geometry.left).toBeGreaterThan(metrics.columns.input.left);
  expect(metrics.columns.geometry.top).toBe(metrics.columns.input.top);
  expect(metrics.columns.input.width).toBeGreaterThanOrEqual(280);
  expect(metrics.columns.geometry.width).toBeGreaterThan(metrics.columns.input.width);
  expect(metrics.columns.input.overflowY).toBe('visible');
  expect(metrics.columns.geometry.overflowY).toBe('visible');
  expect(metrics.columns.output.overflowY).toBe('visible');
  expect(pageErrors).toEqual([]);
});

test('mobile layout uses document flow and never overflows horizontally', async ({ page }) => {
  const pageErrors = capturePageErrors(page);
  await mockBackendServices(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'Photo/Video to STL' })).toBeVisible();
  const metrics = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    clientHeight: document.documentElement.clientHeight,
    scrollHeight: document.documentElement.scrollHeight,
    columns: Array.from(document.querySelectorAll<HTMLElement>('[data-workspace-column]')).map((column) => {
      const rect = column.getBoundingClientRect();
      return { left: rect.left, right: rect.right, width: rect.width, overflowY: getComputedStyle(column).overflowY };
    }),
  }));

  expect(metrics.scrollWidth).toBeLessThanOrEqual(metrics.clientWidth);
  expect(metrics.scrollHeight).toBeGreaterThan(metrics.clientHeight);
  for (const column of metrics.columns) {
    expect(column.left).toBeGreaterThanOrEqual(0);
    expect(column.right).toBeLessThanOrEqual(metrics.clientWidth);
    expect(column.width).toBeGreaterThan(0);
    expect(column.overflowY).toBe('visible');
  }
  expect(pageErrors).toEqual([]);
});

test('run plan is available without dominating the workspace', async ({ page }) => {
  await mockBackendServices(page);
  await page.goto('/');

  const runPlan = page.locator('details').filter({ hasText: 'Run Plan' });
  await expect(runPlan).not.toHaveAttribute('open', '');
  await expect(runPlan.locator('pre')).not.toBeVisible();
  await runPlan.locator('summary').click();
  await expect(runPlan.locator('pre')).toBeVisible();
});

test('print footprint reports independent X and Y dimensions from the media aspect ratio', async ({ page }) => {
  await mockBackendServices(page);
  await page.goto('/');

  await expect(page.getByTestId('print-size-x')).toHaveText('--');
  await expect(page.getByTestId('print-size-y')).toHaveText('--');

  await page.locator('input[type="file"]').setInputFiles({
    name: 'wide.svg',
    mimeType: 'image/svg+xml',
    buffer: Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200"><rect width="400" height="200" fill="white"/></svg>'),
  });

  await expect(page.getByTestId('print-size-x')).toHaveText('256.0 mm');
  await expect(page.getByTestId('print-size-y')).toHaveText('128.0 mm');

  await page.getByLabel('Print size').fill('50');
  await expect(page.getByTestId('print-size-x')).toHaveText('128.0 mm');
  await expect(page.getByTestId('print-size-y')).toHaveText('64.0 mm');
});
