import { defineEval } from "@grok/agent-sdk/evals";

export default defineEval({
  tags: ["smoke"],
  description: "A swing video is measured with analyze-swing instead of guessed.",
  async test(t) {
    await t.send(
      "Analyze examples/shallow-swing.mp4. I am right-handed. Use the analyze-swing tool.",
    );
    t.succeeded();
    t.calledTool("analyze-swing");
  },
});
