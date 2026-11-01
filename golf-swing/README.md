# Golf swing

A down-the-line swing, measured in code, then explained by an agent.

You film from behind, on the target line, at about hand height. The tool draws the address shaft line and the club shaft, saves a video that plays at quarter speed, and says whether the shaft moved down toward that line. That move is shallowing. It is the thing to look at when the downswing stays steep.

One camera behind you cannot measure TrackMan club path, face angle, or clubhead speed. Those are left out on purpose. A steep delivery often goes with an out-to-in path, and the report says so without turning it into a degree number.

## Run it on a video

```bash
pip install -r requirements.txt
# MediaPipe also needs libegl1 on Linux.
python -m swinglab path/to/swing.mp4 -o swing-out
```

`swing-out/swing-analysis.mp4` plays at 0.25x. `swing-out/report.json` has the reading.

```text
reading: shallowed | not_shallowing | steepened | mixed | unclear
```

Positive `abovePlaneDeg` means the middle of the shaft is on the sky side of the address line. That is the steep side. Shallowing is that number falling from the top to P6, the downswing frame where the shaft is closest to horizontal.

`unclear` means the club was not tracked well enough to judge the delivery, usually because the downswing blurred or the clip holds more than one move. The tool will not guess.

Film it this way:

- Phone still, landscape, behind you, hand height, on the target line.
- Whole body and the whole club in frame.
- One swing, from before you move until the finish, at normal speed.
- 60 fps if the phone has it. Right-handed is the default. Pass `--handed left` if you are left-handed.

## Try the drawn swings

```bash
python -m swinglab bot/sandbox/workspace/examples/shallow-swing.mp4 -o /tmp/shallow
python -m swinglab bot/sandbox/workspace/examples/steep-swing.mp4 -o /tmp/steep
python -m unittest tests.test_swing
```

The shallow clip reads `shallowed`. The steep clip reads `not_shallowing`.

## The agent

This folder is an Agent SDK project. `analyze-swing` runs the same measurement. The model only explains the tool result.

```bash
npx @grok/agent-sdk validate --dir .
npx @grok/agent-sdk dev
```

In the playground, ask it to analyze `examples/shallow-swing.mp4`.

Plain Grok chat can watch a video and comment. It will not measure the shaft or draw the plane line. That is why the measurement lives in this tool. A web app can come later. This version is the part you can run on your own clips.

## Where the checks come from

Down-the-line is the view used to see the shaft against the address line. P6 is the downswing frame where the shaft is closest to horizontal, the usual delivery checkpoint ([P system](https://members.wayned.com/basement-tapes-the-p-system-of-golf-swing-positions/), [SwingOK](https://swingok.com/blog/over-the-top-golf-swing)). The plane test is the same drawing a coach puts on a still: extend the address shaft, then see whether the club is above or below that line ([Golf Smart Academy](https://www.golfsmartacademy.com/golf-tips/seeing-club-plane-2d/)). Sportsbox's down-the-line indicators use the same idea, shaft versus a plane, with positive meaning above the plane ([DTL indicators](https://help.sportsbox.ai/what-trackers-and-indicators-are-available-with-dtl-down-the-line), [shaft versus shoulder plane](https://help.sportsbox.ai/swing-plane-trackers)). TrackMan's path and face numbers come from radar, not from one phone camera.

The shaft search is a grip-centered line fit, the same family of idea described by open swing trackers such as [PinPoint Studio](https://github.com/PinPoint-Golf/PinPointStudio). Swing events follow the eight-event sequence from [GolfDB](https://github.com/wmcnally/golfdb). This project does not copy those codebases.
