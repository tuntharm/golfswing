---
description: Use after analyze-swing returns, to explain shallowing and the limits of a down-the-line camera.
---

# Explain a measured swing

1. Say the reading in ordinary words. Quote degrees only when the tool returned them. If `abovePlaneAtTop` is null, skip the degrees.
2. Say the video plays at `playbackSpeed` (0.125x unless they asked for another speed). It shows the clubhead path. Shaft lines are in `elementsPath` and are off the picture.
3. Positive `abovePlaneDeg` means the middle of the shaft is on the sky side of the address shaft line. That is the steep side. Shallowing is that number falling from the top to P6.
4. `shallowed`: the delivery shaft is on or under the address line. If `shallowedDeg` is large, say it dropped by that many degrees from the top. If it is near zero, say the delivery was already on or under the line. Mention one other measured note only if the tool flagged it.
5. `not_shallowing` or `steepened`: the shaft stayed above the address line, or moved further above it. That is the pattern of a swing that is not shallowing. One cue only: from the top, let the clubhead drop behind the hands so the shaft moves down toward the address line before the body turns through.
6. Club path: this camera cannot measure TrackMan path in degrees. A steep delivery often goes with an out-to-in path. That is not a measurement from this video. `clubPathDegrees` is null on purpose.
7. `imageDescentDeg` is only the direction the clubhead moves in the picture. It is not a launch-monitor attack angle. Skip it unless they ask.
8. Tempo near 3:1 is a common full-swing guide, not a pass mark. Mention tempo only when `tempo.ratio` is under 1.8 or over 4.5.
9. `unclear`: if the reason is `clubhead_path_was_marked_on_the_visible_swing`, the video is the saved clubhead path and there is no degree to give. Otherwise the shaft track is not solid enough, often because the downswing club blurred or the clip is long. Ask for a steadier down-the-line clip, hand height, full body, one swing, landscape, normal speed. Do not name a fault.
