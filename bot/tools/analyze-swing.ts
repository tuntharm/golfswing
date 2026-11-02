import { spawn } from "node:child_process";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { prompt } from "@grok/agent-sdk";
import { defineTool } from "@grok/agent-sdk/tools";
import * as z from "zod/v4";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

const Result = z.object({
  reading: z.string(),
  readingReason: z.string(),
  playbackSpeed: z.number(),
  videoPath: z.string().optional(),
  elementsPath: z.string().optional(),
  abovePlaneAtTop: z.number().nullable(),
  abovePlaneAtP6: z.number().nullable(),
  shallowedDeg: z.number().nullable(),
  tempoRatio: z.number().nullable(),
  clubPathDegrees: z.null(),
  warnings: z.array(z.string()),
  swingCount: z.number().optional(),
  swings: z
    .array(
      z.object({
        startSec: z.number(),
        endSec: z.number(),
        reading: z.string().optional(),
        videoPath: z.string().optional(),
      }),
    )
    .optional(),
});

function insideWorkspace(workspaceDir: string, videoPath: string): string {
  const absolute = path.resolve(workspaceDir, videoPath);
  const relative = path.relative(workspaceDir, absolute);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error("videoPath has to stay inside the session workspace.");
  }
  return absolute;
}

export default defineTool({
  description: prompt`
    Measure a down-the-line golf swing and draw the clubhead path at 0.125x.
    Pass playbackSpeed when another speed is requested.
    Shaft lines are saved beside the video and left off the picture.
    Call this when the user supplies a swing video. The numbers in the
    result are the only measurements you may quote. clubPathDegrees is
    always null: one down-the-line camera cannot measure TrackMan path.
  `,
  effect: "write",
  inputSchema: z.object({
    videoPath: z.string().describe("Workspace-relative path to the swing video"),
    handed: z.enum(["right", "left"]).default("right"),
    playbackSpeed: z.number().min(0.1).max(1).default(0.125),
  }),
  outputSchema: Result,
  dryRunResult: {
    reading: "unclear",
    readingReason: "dry_run",
    playbackSpeed: 0.125,
    abovePlaneAtTop: null,
    abovePlaneAtP6: null,
    shallowedDeg: null,
    tempoRatio: null,
    clubPathDegrees: null,
    warnings: ["Dry run. The video was not measured."],
  },
  async execute({ videoPath, handed, playbackSpeed }, ctx) {
    const source = insideWorkspace(ctx.workspaceDir, videoPath);
    await access(source);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const outDir = path.join(ctx.workspaceDir, "swing-out", stamp);
    const output = await runAnalyzer(source, outDir, handed, playbackSpeed);
    const report = JSON.parse(await readFile(path.join(outDir, "report.json"), "utf8")) as Report;
    const result = shape(report, playbackSpeed);
    const videos = new Set<string>();
    if (output.videoPath) videos.add(output.videoPath);
    for (const swing of result.swings ?? []) {
      if (swing.videoPath) videos.add(swing.videoPath);
    }
    let index = 0;
    for (const file of videos) {
      const video = await readFile(file);
      await ctx.artifacts.tag({
        kind: "swing-report",
        key: `${ctx.session.id}:${path.basename(videoPath)}:${playbackSpeed}:${index}`,
        title: `Swing ${result.reading} ${index + 1}`,
        data: {
          reading: result.reading,
          playbackSpeed,
          abovePlaneAtP6: result.abovePlaneAtP6,
        },
        contents: video,
        contentType: "video/mp4",
      });
      index += 1;
    }
    return result;
  },
});

type Report = {
  reading?: string;
  readingReason?: string;
  playbackSpeed?: number;
  videoPath?: string;
  elementsPath?: string;
  warnings?: string[];
  swingCount?: number;
  swings?: { startSec: number; endSec: number; reading?: string; videoPath?: string }[];
  shallowing?: {
    abovePlaneDeg?: { top?: number | null; p6?: number | null };
    shallowedDeg?: number | null;
  };
  tempo?: { ratio?: number | null };
  delivery?: { clubPathDegrees?: null };
};

function shape(report: Report, playbackSpeed: number) {
  return {
    reading: report.reading ?? "unclear",
    readingReason: report.readingReason ?? "no_reading",
    playbackSpeed: report.playbackSpeed ?? playbackSpeed,
    videoPath: report.videoPath,
    elementsPath: report.elementsPath,
    abovePlaneAtTop: report.shallowing?.abovePlaneDeg?.top ?? null,
    abovePlaneAtP6: report.shallowing?.abovePlaneDeg?.p6 ?? null,
    shallowedDeg: report.shallowing?.shallowedDeg ?? null,
    tempoRatio: report.tempo?.ratio ?? null,
    clubPathDegrees: null,
    warnings: report.warnings ?? [],
    swingCount: report.swingCount,
    swings: report.swings,
  };
}

function runAnalyzer(
  source: string,
  outDir: string,
  handed: string,
  playbackSpeed: number,
): Promise<{ videoPath?: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      "python3",
      [
        "-m",
        "swinglab",
        source,
        "-o",
        outDir,
        "--handed",
        handed,
        "--speed",
        String(playbackSpeed),
      ],
      { cwd: projectRoot },
    );
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("Swing measurement took too long."));
    }, 180_000);
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(stderr.trim() || `swing measurement exited ${code}`));
        return;
      }
      resolve({ videoPath: path.join(outDir, "swing-analysis.mp4") });
    });
  });
}
