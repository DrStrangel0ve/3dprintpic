import { expect, test } from '@playwright/test';

const onePixelPng = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z5WQAAAAASUVORK5CYII=',
  'base64',
);
const originalPng = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGO8I2LDwMDAxAAGAA7aATAS/mzBAAAAAElFTkSuQmCC',
  'base64',
);
const selectedPng = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGOUm/CfgYGBiQEMABNlAbHv481xAAAAAElFTkSuQmCC',
  'base64',
);

const tetrahedronStl = Buffer.from(`solid tetrahedron
facet normal 0 0 -1
 outer loop
  vertex 0 0 0
  vertex 0 1 0
  vertex 1 0 0
 endloop
endfacet
facet normal 0 -1 0
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 0 1
 endloop
endfacet
facet normal -1 0 0
 outer loop
  vertex 0 0 0
  vertex 0 0 1
  vertex 0 1 0
 endloop
endfacet
facet normal 1 1 1
 outer loop
  vertex 1 0 0
  vertex 0 1 0
  vertex 0 0 1
 endloop
endfacet
endsolid tetrahedron
`);

test('selected relief sends the selected preview and atomic compose job', async ({ page }) => {
  let composeMultipartBody = '';
  let processMultipartBody = '';
  let processMultipartBuffer = Buffer.alloc(0);
  let precomputedMaskMultipartBody = '';
  let liveSelectionMaskRequests = 0;
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
        defaults: {
          selection: 'sam3-person-aware',
          frame_selection: 'uniform-frame-sampler',
          camera_pose: 'turntable-orbit',
          video_reconstruction: 'multiview-visual-hull',
          image_to_mesh: 'triposg',
          stl_postprocess: 'trimesh-repair',
        },
        groups: {
          selection: [{ id: 'sam3-person-aware', label: 'SAM 3 Person-aware', availability: 'configured' }],
          frame_selection: [{ id: 'uniform-frame-sampler', label: 'Uniform frames' }],
          camera_pose: [{ id: 'turntable-orbit', label: 'Turntable orbit' }],
          video_reconstruction: [{ id: 'multiview-visual-hull', label: 'Visual hull' }],
          image_to_mesh: [{ id: 'triposg', label: 'TripoSG' }],
          stl_postprocess: [{ id: 'trimesh-repair', label: 'Trimesh repair' }],
        },
      }),
    }),
  );
  await page.route('**/providers/image-to-mesh', (route) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ providers: [] }) }),
  );
  await page.route('**/selection/precompute', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        precompute_supported: true,
        precompute_id: 'sam3-cache-1',
        model_status: 'sam3-concepts-precomputed',
        segment_count: 3,
      }),
    }),
  );
  await page.route('**/selection/precomputed_mask', (route) => {
    precomputedMaskMultipartBody = (route.request().postDataBuffer() || Buffer.alloc(0)).toString('utf8');
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        mask: 'selection/preview/mask.png',
        mask_url: '/selection-assets/mask.png',
        tint_url: '/selection-assets/tint.png',
        model_status: 'sam3-concept-precomputed-point',
      }),
    });
  });
  await page.route('**/selection/mask', (route) => {
    liveSelectionMaskRequests += 1;
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        mask: 'selection/preview/mask.png',
        mask_url: '/selection-assets/mask.png',
        tint_url: '/selection-assets/tint.png',
        model_status: 'ready',
      }),
    });
  });
  await page.route('**/selection/compose', (route) => {
    composeMultipartBody = (route.request().postDataBuffer() || Buffer.alloc(0)).toString('utf8');
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        job_id: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        selected_image: 'selection/composed/selected_image.png',
        selected_image_url: '/selection-assets/selected_image.png',
        mask: 'selection/composed/selection_mask.png',
        mask_url: '/selection-assets/composed_mask.png',
        mask_coverage: 0.42,
        model_status: 'composed-clicked-masks',
      }),
    });
  });
  await page.route('**/selection-assets/*.png', (route) =>
    route.fulfill({
      contentType: 'image/png',
      body: route.request().url().includes('selected_image') ? selectedPng : onePixelPng,
    }),
  );
  await page.route('**/process_image', async (route) => {
    processMultipartBuffer = route.request().postDataBuffer() || Buffer.alloc(0);
    processMultipartBody = processMultipartBuffer.toString('utf8');
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        stl_url: '/artifacts/selection-context/output_model.stl',
        diagnostics_url: '/artifacts/selection-context/diagnostics.json',
        runtime: { device: 'Test GPU', cuda_available: true },
        depth_metadata: { effective_model: 'depth-anything/Depth-Anything-V2-Large-hf' },
        selection_depth_context: { enabled: true },
        stl_diagnostics: {
          stl_exists: true,
          stl_is_watertight: true,
          stl_is_volume: true,
          stl_is_manifold: true,
          stl_winding_consistent: true,
          stl_positive_volume: true,
          stl_single_component: true,
          stl_passes_hard_checks: true,
          stl_faces: 4,
        },
      }),
    });
  });
  await page.route('**/artifacts/selection-context/output_model.stl', (route) =>
    route.fulfill({ contentType: 'model/stl', body: tetrahedronStl }),
  );

  await page.goto('/');
  await page.getByRole('button', { name: 'Full Mesh STL' }).click();
  await expect(page.getByText('Inpainting', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: '2.5D Relief STL' }).click();
  await page.locator('input[type="file"]').setInputFiles({
    name: 'llama.png',
    mimeType: 'image/png',
    buffer: originalPng,
  });
  await page.getByRole('button', { name: 'Object selection' }).click();
  await expect(page.getByText('Precomputed 3 segments', { exact: true })).toBeVisible();
  const selectionImage = page.locator('img.cursor-crosshair');
  await expect(selectionImage).toBeVisible();
  await selectionImage.click({ position: { x: 1, y: 1 } });
  await expect(page.getByRole('button', { name: 'Apply' })).toBeEnabled();
  await page.getByRole('button', { name: 'Apply' }).click();
  await expect(page.getByText('Selection preview', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Run' }).click();
  await expect(page.getByText('STL ready', { exact: true })).toBeVisible();

  expect(composeMultipartBody).toContain('name="selection_infill_mode"');
  expect(composeMultipartBody).toContain('none');
  expect(precomputedMaskMultipartBody).toContain('name="precompute_id"');
  expect(precomputedMaskMultipartBody).toContain('sam3-cache-1');
  expect(liveSelectionMaskRequests).toBe(0);
  expect(processMultipartBody).toContain('name="file"; filename="selected-llama.png"');
  expect(processMultipartBody).toContain('name="selection_job_id"');
  expect(processMultipartBody).toContain('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
  expect(processMultipartBody).toContain('name="selection_mode"');
  expect(processMultipartBody).toContain('context');
  expect(processMultipartBody).not.toContain('name="completion_mode"');
  expect(processMultipartBody).not.toContain('name="selection_background_depth_ratio"');
  expect(processMultipartBody).not.toContain('isolate');
  expect(processMultipartBody).toContain('name="selection_subject_lock"');
  expect(processMultipartBody).toContain('true');
  expect(processMultipartBody).toContain('name="depth_downsample_sharpening"');
  expect(processMultipartBody).toContain('0.35');
  expect(processMultipartBody).not.toContain('name="depth_context_file"');
  expect(processMultipartBuffer.includes(selectedPng)).toBe(true);
  expect(processMultipartBuffer.includes(originalPng)).toBe(false);
});

test('late compose response cannot attach an old mask to a replacement photo', async ({ page }) => {
  let releaseCompose: (() => void) | undefined;
  const composeGate = new Promise<void>((resolve) => {
    releaseCompose = resolve;
  });
  let processRequests = 0;

  await page.route('**/health', (route) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ status: 'ok' }) }),
  );
  await page.route('**/depth/preload/depthpro/status', (route) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ status: 'missing' }) }),
  );
  await page.route('**/models', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        defaults: {
          selection: 'sam3-person-aware',
          frame_selection: 'uniform-frame-sampler',
          camera_pose: 'turntable-orbit',
          video_reconstruction: 'multiview-visual-hull',
          image_to_mesh: 'triposg',
          stl_postprocess: 'trimesh-repair',
        },
        groups: {
          selection: [{ id: 'sam3-person-aware', label: 'SAM 3 Person-aware' }],
          frame_selection: [{ id: 'uniform-frame-sampler', label: 'Uniform frames' }],
          camera_pose: [{ id: 'turntable-orbit', label: 'Turntable orbit' }],
          video_reconstruction: [{ id: 'multiview-visual-hull', label: 'Visual hull' }],
          image_to_mesh: [{ id: 'triposg', label: 'TripoSG' }],
          stl_postprocess: [{ id: 'trimesh-repair', label: 'Trimesh repair' }],
        },
      }),
    }),
  );
  await page.route('**/providers/image-to-mesh', (route) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ providers: [] }) }),
  );
  await page.route('**/selection/precompute', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ precompute_supported: false, model_status: 'live-mask' }),
    }),
  );
  await page.route('**/selection/mask', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        mask: 'selection/first/mask.png',
        mask_url: '/selection-assets/mask.png',
        tint_url: '/selection-assets/tint.png',
      }),
    }),
  );
  await page.route('**/selection-assets/*.png', (route) =>
    route.fulfill({ contentType: 'image/png', body: onePixelPng }),
  );
  await page.route('**/selection/compose', async (route) => {
    await composeGate;
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        job_id: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
        selected_image: 'selection/first/selected_image.png',
        selected_image_url: '/selection-assets/selected_image.png',
        mask: 'selection/first/selection_mask.png',
      }),
    });
  });
  await page.route('**/process_image', (route) => {
    processRequests += 1;
    return route.fulfill({ status: 500, body: 'stale request should not run' });
  });

  await page.goto('/');
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles({ name: 'first.png', mimeType: 'image/png', buffer: originalPng });
  await page.getByRole('button', { name: 'Object selection' }).click();
  await page.locator('img.cursor-crosshair').click({ position: { x: 1, y: 1 } });
  await page.getByRole('button', { name: 'Apply' }).click();
  await expect(page.getByText('Applying object selection', { exact: true })).toBeVisible();

  await fileInput.setInputFiles({ name: 'second.png', mimeType: 'image/png', buffer: selectedPng });
  releaseCompose?.();
  await expect(page.getByText('Ready', { exact: true })).toBeVisible();
  await expect(page.getByText('Selection preview', { exact: true })).toHaveCount(0);

  await page.getByRole('button', { name: 'Run' }).click();
  await expect(page.getByText('Selection required', { exact: true })).toBeVisible();
  expect(processRequests).toBe(0);
});
