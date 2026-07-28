import { expect, test } from '@playwright/test';

test('video Run submits the live turntable pipeline and exposes its STL', async ({ page }) => {
  const pageErrors: string[] = [];
  let multipartBody = '';
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.route('**/health', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ status: 'ok', runtime: { device: 'Test GPU', cuda_available: true } }),
    }),
  );
  await page.route('**/depth/preload/depthpro/status', (route) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ status: 'missing' }) }),
  );
  await page.route('**/models', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        service: 'video-runner-test',
        mode: 'planner-plus-runner',
        defaults: {
          selection: 'detr-resnet-50-panoptic',
          frame_selection: 'uniform-frame-sampler',
          camera_pose: 'turntable-orbit',
          video_reconstruction: 'multiview-visual-hull',
          image_to_mesh: 'triposg',
          stl_postprocess: 'trimesh-repair',
        },
        groups: {
          selection: [
            { id: 'turntable-grabcut', label: 'Turntable foreground', availability: 'configured' },
            { id: 'detr-resnet-50-panoptic', label: 'DETR Panoptic', availability: 'configured' },
          ],
          frame_selection: [
            { id: 'uniform-frame-sampler', label: 'Uniform frame sampler', availability: 'configured' },
          ],
          camera_pose: [{ id: 'turntable-orbit', label: 'Turntable orbit', availability: 'configured' }],
          video_reconstruction: [
            { id: 'multiview-visual-hull', label: 'Multiview visual hull', availability: 'configured' },
          ],
          image_to_mesh: [{ id: 'triposg', label: 'TripoSG', availability: 'setup-required' }],
          stl_postprocess: [{ id: 'trimesh-repair', label: 'Trimesh repair', availability: 'configured' }],
        },
        metrics: ['watertightness', 'manifoldness'],
      }),
    }),
  );
  await page.route('**/providers/image-to-mesh', (route) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ providers: [] }) }),
  );
  await page.route('**/plan', async (route) => {
    const request = route.request();
    const payload = request.postDataJSON();
    expect(payload.input.media_type).toBe('video');
    expect(payload.models.camera_pose).toBe('turntable-orbit');
    expect(payload.models.video_reconstruction).toBe('multiview-visual-hull');
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ status: 'planned', service: 'video-runner-test', execution_mode: 'planner-only' }),
    });
  });
  await page.route('**/run/video-to-mesh', async (route) => {
    multipartBody = route.request().postDataBuffer()?.toString('utf8') || '';
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'printable',
        stl_passes_hard_checks: true,
        stl_failed_checks: [],
        stl_url: '/artifacts/video-job/output_model.stl',
        diagnostics_url: '/artifacts/video-job/diagnostics.json',
        selected_frame_urls: ['/artifacts/video-job/prepared/frames/view_000.png'],
        mask_quality: { passes_hard_checks: true },
        stl_diagnostics: {
          runner: 'video-to-mesh',
          stl_exists: true,
          stl_is_watertight: true,
          stl_is_volume: true,
          stl_is_manifold: true,
          stl_winding_consistent: true,
          stl_positive_volume: true,
          stl_single_component: true,
        },
        timings: { preparation_seconds: 0.5, provider_seconds: 0.2, total_seconds: 0.8 },
      }),
    });
  });
  await page.route('**/artifacts/**', (route) =>
    route.fulfill({
      contentType: 'image/png',
      body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z5WQAAAAASUVORK5CYII=', 'base64'),
    }),
  );

  await page.goto('/');
  await page.locator('input[type="file"]').setInputFiles({
    name: 'turntable.avi',
    mimeType: 'video/x-msvideo',
    buffer: Buffer.from('synthetic-video'),
  });
  await expect(page.getByRole('button', { name: 'Video' })).toHaveClass(/border-blue-700/);
  await expect(page.getByLabel('Selection model')).toHaveValue('turntable-grabcut');

  await page.getByRole('button', { name: 'Run' }).click();

  await expect(page.getByText('Printable video STL ready')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Download STL' })).toHaveAttribute(
    'href',
    'http://localhost:8005/artifacts/video-job/output_model.stl',
  );
  expect(multipartBody).toContain('name="frame_selection"');
  expect(multipartBody).toContain('uniform-frame-sampler');
  expect(multipartBody).toContain('name="segmentation_provider"');
  expect(multipartBody).toContain('turntable-grabcut');
  expect(multipartBody).toContain('name="selected_frame_count"');
  expect(multipartBody).toContain('12');
  expect(pageErrors).toEqual([]);
});
