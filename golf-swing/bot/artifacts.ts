import { defineArtifacts } from "@grok/agent-sdk/artifacts";
import * as z from "zod/v4";

export default defineArtifacts({
  kinds: {
    "swing-report": {
      description: "Quarter-speed swing video with the address plane and shaft drawn on it.",
      schema: z.object({
        reading: z.string(),
        playbackSpeed: z.number(),
        abovePlaneAtP6: z.number().nullable(),
      }),
    },
  },
});
