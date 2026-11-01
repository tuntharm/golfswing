import { defineArtifacts } from "@grok/agent-sdk/artifacts";
import * as z from "zod/v4";

export default defineArtifacts({
  kinds: {
    "swing-report": {
      description: "Quarter-speed swing video. The clubhead path is drawn. Shaft lines are left off the picture.",
      schema: z.object({
        reading: z.string(),
        playbackSpeed: z.number(),
        abovePlaneAtP6: z.number().nullable(),
      }),
    },
  },
});
