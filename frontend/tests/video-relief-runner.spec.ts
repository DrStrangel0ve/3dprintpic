import { expect, test } from '@playwright/test';

test('tracked video relief submits the non-turntable runner and exposes its STL', async ({ page }) => {
  let reliefBody = '';
  let meshCalled = false;

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
        service: 'video-relief-test',
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
            { id: 'sam2.1-hiera-tiny-video', label: 'SAM 2.1 Tiny Video', availability: 'configured' },
            { id: 'detr-resnet-50-panoptic', label: 'DETR Panoptic', availability: 'configured' },
          ],
          frame_selection: [
            { id: 'uniform-frame-sampler', label: 'Uniform frame sampler', availability: 'configured' },
            { id: 'sharpness-motion-selector', label: 'Sharpness/motion selector', availability: 'configured' },
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
    const payload = route.request().postDataJSON();
    expect(payload.input.media_type).toBe('video');
    expect(payload.route.target).toBe('tracked-relief');
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ status: 'planned', service: 'video-relief-test', execution_mode: 'runner' }),
    });
  });
  await page.route('**/run/video-to-mesh', async (route) => {
    meshCalled = true;
    await route.abort();
  });
  await page.route('**/run/video-to-relief', async (route) => {
    reliefBody = route.request().postDataBuffer()?.toString('utf8') || '';
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'printable',
        stl_passes_hard_checks: true,
        stl_failed_checks: [],
        stl_url: '/artifacts/relief-job/output_model.stl',
        diagnostics_url: '/artifacts/relief-job/diagnostics.json',
        preview_url: '/artifacts/relief-job/selected_subject.png',
        selected_frame_url: '/artifacts/relief-job/selected_frame.png',
        motion: { classification: 'approach' },
        stl_diagnostics: {
          runner: 'video-to-relief',
          stl_exists: true,
          stl_is_watertight: true,
          stl_is_volume: true,
          stl_is_manifold: true,
          stl_winding_consistent: true,
          stl_positive_volume: true,
          stl_single_component: true,
        },
        timings: { tracking_seconds: 1.0, depth_seconds: 0.5, stl_seconds: 0.2, total_seconds: 1.7 },
      }),
    });
  });
  await page.route('**/artifacts/**', (route) =>
    route.fulfill({
      contentType: 'image/png',
      body: Buffer.from(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z5WQAAAAASUVORK5CYII=',
        'base64',
      ),
    }),
  );

  await page.goto('/');
  await page.locator('input[type="file"]').setInputFiles({
    name: 'aarusnow.mp4',
    mimeType: 'video/mp4',
    buffer: Buffer.from('synthetic-video'),
  });
  await page.getByRole('button', { name: 'Tracked relief' }).click();
  await expect(page.getByLabel('Frame selector')).toHaveValue('sharpness-motion-selector');
  await expect(page.getByLabel('Selection model')).toHaveValue('sam2.1-hiera-tiny-video');

  await page.getByRole('button', { name: 'Run' }).click();

  await expect(page.getByText('Printable tracked relief ready')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Download STL' })).toHaveAttribute(
    'href',
    'http://localhost:8005/artifacts/relief-job/output_model.stl',
  );
  expect(meshCalled).toBe(false);
  expect(reliefBody).toContain('name="segmentation_provider"');
  expect(reliefBody).toContain('sam2.1-hiera-tiny-video');
  expect(reliefBody).toContain('name="frame_selection"');
  expect(reliefBody).toContain('sharpness-motion-selector');
  expect(reliefBody).toContain('name="depth_model"');
  expect(reliefBody).toContain('Depth-Anything-V2-Large-hf');
});
