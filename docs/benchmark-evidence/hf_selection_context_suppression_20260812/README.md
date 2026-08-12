# Hosted selection context suppression

## Failure

The cached SAM 3 hover map resolved overlapping masks using raw confidence
alone. In crowded travel photos, a high-confidence building or vegetation mask
could span most of the frame and either highlight the background or win a
foreground click. The transient hover region also remained painted immediately
after committing a selection.

## Correction

- Non-person masks covering at least 55% of the image are excluded from the
  selectable hit map.
- Non-person masks covering at least 25% and touching three or more frame sides
  are likewise treated as scene context.
- Remaining overlaps use confidence minus a bounded area-specificity penalty,
  allowing a smaller foreground object to win when scores are close.
- A committed click clears transient hover while retaining all selected-object
  highlights.

Only hit testing changes. Accepted object masks are still the original packed
SAM 3 pixels, and selecting or composing objects requires no additional GPU
inference.

## Deterministic regression

A synthetic four-mask fixture combines a 99%-confidence frame-spanning building
context mask, a person, a furniture object, and a broad overlapping furniture
mask. The context pixels become non-selectable, the person wins its overlap,
and the independent furniture object remains selectable. Existing disconnected
component, packed-session, multi-click, and end-to-end selected-relief tests
remain required.

No uploaded image, source mask, or private selection artifact is stored here.
