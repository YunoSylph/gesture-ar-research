# Local Gesture Reference Clips

These clips are duplicated from the IPN Hand subset and define the exact gesture semantics for local recording.
Record local clips with the same target labels and the same gesture meaning. Do not introduce new gesture variants.

Target labels:

- `no_gesture`: absence of command; IPN source class `No gesture`; interaction `suppress/reset`.
- `point_2f`: two-finger pointing; IPN source class `Point-2f`; interaction `pointer/hover`.
- `click_2f`: two-finger click; IPN source class `Click-2f`; interaction `select/confirm`.
- `swipe_left`: leftward motion; IPN source class `Th-left`; interaction `previous/rotate left`.
- `swipe_right`: rightward motion; IPN source class `Th-right`; interaction `next/rotate right`.
- `zoom_in`: pinch/open zoom in; IPN source class `Zoom-in`; interaction `transform zoom in`.
- `zoom_out`: pinch/close zoom out; IPN source class `Zoom-o`; interaction `transform zoom out`.

Generated clips:

- `no_gesture`: `no_gesture\no_gesture_ref_01.mp4` from `1CV12_13_R_#95_000001_000067_D0X`
- `no_gesture`: `no_gesture\no_gesture_ref_02.mp4` from `4CM11_13_R_#29_001349_001415_D0X`
- `no_gesture`: `no_gesture\no_gesture_ref_03.mp4` from `1CM1_2_R_#223_003422_003489_D0X`
- `point_2f`: `point_2f\point_2f_ref_01.mp4` from `4CM11_16_R_#210_000184_000250_B0B`
- `point_2f`: `point_2f\point_2f_ref_02.mp4` from `1CM42_12_R_#158_004395_004476_B0B`
- `point_2f`: `point_2f\point_2f_ref_03.mp4` from `1CV12_8_R_#88_003640_003745_B0B`
- `click_2f`: `click_2f\click_2f_ref_01.mp4` from `1CV12_13_R_#94_002232_002298_G02`
- `click_2f`: `click_2f\click_2f_ref_02.mp4` from `4CM11_20_R_#42_002835_002902_G02`
- `click_2f`: `click_2f\click_2f_ref_03.mp4` from `1CM42_17_R_#191_003236_003299_G02`
- `swipe_left`: `swipe_left\swipe_left_ref_01.mp4` from `1CM42_21_R_#155_004138_004204_G05`
- `swipe_left`: `swipe_left\swipe_left_ref_02.mp4` from `1CV12_13_R_#96_004463_004529_G05`
- `swipe_left`: `swipe_left\swipe_left_ref_03.mp4` from `4CM11_16_R_#209_002576_002642_G05`
- `swipe_right`: `swipe_right\swipe_right_ref_01.mp4` from `1CM42_11_R_#205_002546_002612_G06`
- `swipe_right`: `swipe_right\swipe_right_ref_02.mp4` from `1CM1_1_R_#219_001501_001566_G06`
- `swipe_right`: `swipe_right\swipe_right_ref_03.mp4` from `4CM11_16_R_#209_000388_000453_G06`
- `zoom_in`: `zoom_in\zoom_in_ref_01.mp4` from `4CM11_20_R_#44_003977_004043_G10`
- `zoom_in`: `zoom_in\zoom_in_ref_02.mp4` from `1CM1_3_R_#225_003858_003923_G10`
- `zoom_in`: `zoom_in\zoom_in_ref_03.mp4` from `1CV12_23_R_#119_002768_002833_G10`
- `zoom_out`: `zoom_out\zoom_out_ref_01.mp4` from `1CV12_15_R_#103_001677_001743_G11`
- `zoom_out`: `zoom_out\zoom_out_ref_02.mp4` from `4CM11_16_R_#211_000506_000572_G11`
- `zoom_out`: `zoom_out\zoom_out_ref_03.mp4` from `1CM42_26_R_#173_003742_003809_G11`

Recommended local capture: 5-10 clips per target label, 2-4 seconds each, one gesture per clip.
Keep the hand fully visible and use the same label names as above.
