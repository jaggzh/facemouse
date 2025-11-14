Calibration needs to reach all the way to the edges of the screen, not just a neat little grid in the middle, because the user actually needs accurate cursor control in the corners and along the borders, so the UI has to be able to draw targets flush to all edges in fullscreen, without any “safe margins” eating into that space.

Calibration shouldn’t be an all-or-nothing ritual every time; we want both quick coarse modes, like a fast four-corner or 3×3 grid, and refinement passes that let the user fix only the area that’s misbehaving, like “top left is off, let’s correct just that region” instead of forcing a full restart.

Manual calibration is a first-class tool, not a fallback, where the user moves the actual mouse, clicks near a grid point, that point becomes the active target, and a second confirming click near the same spot commits the gaze sample, with the system automatically snapping to the nearest calibration point and showing a simple target (dot in a circle, no crosshairs that might confuse the eyes).

The patient’s physical constraints shape everything: limited head movement, a ventilator and CPAP mask that cover parts of the face, and occasional partial occlusion of one or both eyes mean the tracker must tolerate frequent, brief failures, skip updates when recognition fails rather than injecting garbage, and gracefully fall back to one-eye tracking when needed.

Mediapipe is the backbone of face and eye detection, because in practice it has been the only system that reliably finds this patient’s face, so the design assumes a continuous stream of landmarks with gaps, uses both eyes when available for better accuracy, and distinguishes between long-term “one-eye mode” and short-term “one eye dropped out for a moment” so the filters and smoothing behave differently in each situation.

Cursor control should feel supportive rather than twitchy, so the gaze-to-cursor mapping will likely include behavior that slows and stabilizes the cursor as motion decreases, almost freezing when the user “settles” on a point, yet still allowing small head or eye adjustments to nudge the cursor into a fine position without constant jitter.

Visualization is part of the UX, not just debug output; live video with overlaid features, gaze points, and fading trails can make jumpiness, lag, and accuracy immediately visible, can be fun and motivating for the patient, and can also double as a calibration and diagnostics surface, especially when combined with on-screen prompts and TTS cues.

Configuration and behavior need to be tweakable and transparent, so settings like smoothing strength, filter toggles, mode thresholds (two-eye, one-eye, one-drop), and calibration data will live in a YAML config under the user’s config directory, easy to read and hand-edit, while the UI exposes the most important controls without overwhelming the user.
