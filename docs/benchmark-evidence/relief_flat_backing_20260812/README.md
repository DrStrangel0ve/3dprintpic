# Flat relief backing correction

## Failure

The local UI exposed a 2.4 mm `Base thickness` control, but the relief request
did not submit it and `/process_image` did not accept it. The height-field
exporter therefore retained its original 0.01 mm numerical buffer as the entire
backing thickness. At an oblique viewing angle that nearly coplanar shell could
appear as a thin triangular extrusion along the lower edge. It was also below a
useful printable backing thickness.

## Correction

- `/process_image` accepts `base_thickness_mm` in the bounded 0.4-20 mm range
  and defaults to the existing local UI value of 2.4 mm.
- Local and hosted clients both submit the same base-thickness value.
- The approved relief field is translated upward by that constant thickness;
  its gradients, depth span, face shape, background detail, and attachment
  geometry are unchanged.
- The backing closes to one exact `Z=0` plane. The exporter records
  `flat_backing_plane_v1` provenance and the bottom/top backing planes.
- Direct historical benchmark calls keep the legacy 0.01 mm default unless
  they explicitly opt into the production backing value, preserving replay
  semantics.

## Measured gates

The focused deterministic fixture compares identical reliefs at 0.01 mm and
2.4 mm backing thickness with border flattening both disabled and enabled.
After subtracting their constant offsets, every front-surface sample agrees
within `2e-6` mm. The 2.4 mm mesh has:

- minimum mesh Z exactly `0`;
- minimum front-surface Z exactly `2.4` mm;
- zero vertices strictly between the bottom plane and backing top;
- one watertight, consistently wound positive-volume component;
- unchanged topological face count relative to the legacy fixture.

The selected/full-scene Space regression additionally requires matching valid
surface masks and identical face counts, so object selection cannot reintroduce
the former mask-shaped holes while using the corrected backing.

No private source image, mask, depth map, or STL is included here.
