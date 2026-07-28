import { expect, test } from '@playwright/test';


const onePixelPng = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z5WQAAAAASUVORK5CYII=',
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

test('single-photo scene route emits orbitable STL and colored GLB artifacts', async ({ page }) => {
  const pageErrors: string[] = [];
  let sceneMultipartBody = '';
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
        service: 'scene-test',
        mode: 'runner',
        defaults: {
          selection: 'detr-resnet-50-panoptic',
          frame_selection: 'uniform-frame-sampler',
          camera_pose: 'turntable-orbit',
          video_reconstruction: 'multiview-visual-hull',
          image_to_mesh: 'triposg',
          stl_postprocess: 'trimesh-repair',
        },
        groups: {
          selection: [{ id: 'detr-resnet-50-panoptic', label: 'DETR Panoptic', availability: 'configured' }],
          frame_selection: [{ id: 'uniform-frame-sampler', label: 'Uniform frames' }],
          camera_pose: [{ id: 'turntable-orbit', label: 'Turntable orbit' }],
          video_reconstruction: [{ id: 'multiview-visual-hull', label: 'Visual hull' }],
          image_to_mesh: [{ id: 'triposg', label: 'TripoSG' }],
          stl_postprocess: [{ id: 'trimesh-repair', label: 'Trimesh repair' }],
        },
        metrics: ['watertightness'],
      }),
    }),
  );
  await page.route('**/providers/image-to-mesh', (route) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ providers: [] }) }),
  );
  await page.route('**/process_scene_diorama', async (route) => {
    sceneMultipartBody = route.request().postDataBuffer()?.toString('utf8') || '';
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        stl_url: '/artifacts/scene-job/output_scene.stl',
        scene_url: '/artifacts/scene-job/output_scene.glb',
        preview_url: '/artifacts/scene-job/output_scene_preview.png',
        diagnostics_url: '/artifacts/scene-job/diagnostics.json',
        runtime: { device: 'Test GPU', cuda_available: true },
        depth_metadata: { effective_model: 'depth-anything/Depth-Anything-V2-Large-hf' },
        timings: { depth_seconds: 0.2, scene_mesh_seconds: 0.3, total_seconds: 0.6 },
        stl_diagnostics: {
          runner: 'single-photo-scene-diorama',
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
  await page.route('**/artifacts/scene-job/output_scene.stl', (route) =>
    route.fulfill({ contentType: 'model/stl', body: tetrahedronStl }),
  );
  await page.route('**/artifacts/scene-job/output_scene_preview.png', (route) =>
    route.fulfill({ contentType: 'image/png', body: onePixelPng }),
  );

  await page.goto('/');
  await page.locator('input[type="file"]').setInputFiles({
    name: 'selfie.png',
    mimeType: 'image/png',
    buffer: onePixelPng,
  });
  await page.getByRole('button', { name: 'Scene Diorama' }).click();
  await expect(page.getByLabel('Scene depth model')).toBeVisible();
  await page.getByRole('button', { name: 'Whole image' }).click();
  await page.getByRole('button', { name: 'Run' }).click();

  await expect(page.getByText('Scene GLB and STL ready')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Download STL' })).toHaveAttribute(
    'href',
    /\/artifacts\/scene-job\/output_scene\.stl$/,
  );
  await expect(page.getByRole('link', { name: 'Download Scene GLB' })).toHaveAttribute(
    'href',
    /\/artifacts\/scene-job\/output_scene\.glb$/,
  );
  await expect(page.getByTestId('stl-orbit-viewer')).toBeVisible();
  const outputCanvas = page.locator('#stl-output-canvas canvas');
  await expect(outputCanvas).toBeVisible();
  await page.waitForTimeout(500);
  const canvasPixels = await outputCanvas.evaluate((node) => {
    const canvas = node as HTMLCanvasElement;
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return { width: 0, height: 0, channelRange: 0, nonZeroPixels: 0 };
    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let minimum = 255;
    let maximum = 0;
    let nonZeroPixels = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      const red = pixels[index];
      const green = pixels[index + 1];
      const blue = pixels[index + 2];
      minimum = Math.min(minimum, red, green, blue);
      maximum = Math.max(maximum, red, green, blue);
      if (red || green || blue) nonZeroPixels += 1;
    }
    return { width, height, channelRange: maximum - minimum, nonZeroPixels };
  });
  const viewerScreenshot = await page.getByTestId('stl-orbit-viewer').screenshot();
  expect(canvasPixels.width).toBeGreaterThan(100);
  expect(canvasPixels.height).toBeGreaterThan(100);
  expect(canvasPixels.channelRange).toBeGreaterThan(20);
  expect(canvasPixels.nonZeroPixels).toBeGreaterThan(100);
  expect(viewerScreenshot.byteLength).toBeGreaterThan(1000);

  expect(sceneMultipartBody).toContain('name="mask_paths_json"');
  expect(sceneMultipartBody).toContain('name="scene_depth_mm"');
  expect(sceneMultipartBody).toContain('name="facade_detail_mm"');
  expect(sceneMultipartBody).toContain('name="subject_depth_mm"');
  expect(pageErrors).toEqual([]);
});
