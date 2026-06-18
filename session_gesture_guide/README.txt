Gesture session guide - Object control task
============================================

Open index.html in a browser for the animated cheat-sheet, or use the files
in order. Perform one gesture at a time and return the hand to a neutral,
relaxed open pose for ~2 seconds between steps.

Order:
  1. Point      (1_point.mp4)          -> pointer_hover    : two-finger point, hold briefly
  2. Click      (2_click.mp4)          -> select_confirm   : two-finger click (index + middle)
  3. Zoom in    (3_zoom_in_pinch.svg)  -> zoom_in          : spread thumb + index apart (pinch open)
  4. Zoom out   (4_zoom_out_pinch.svg) -> zoom_out         : bring thumb + index together (pinch close)

Recording:
  - App: Webcam + Robust C6 + Object control task -> Start Task.
  - Do steps 1 -> 2 -> 3 -> 4, then stop.
  - Repeat for 3-5 separate sessions.
  - Face a light source and keep the whole hand in frame at medium distance;
    on earlier logs half the frames lost the hand.

Note: steps 1-2 are real IPN reference clips; steps 3-4 show the thumb-index
pinch the live controller actually uses (the dataset zoom clip moved the whole
hand and no longer matches the controller).
