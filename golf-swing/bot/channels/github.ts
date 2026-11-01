import { defaultGitHubAuth, defaultOnComment, githubChannel } from "@grok/agent-sdk/channels/github";

const wakeOn = new Set(["opened", "reopened", "synchronize", "ready_for_review"]);

export default githubChannel({
  botName: "golf-swing",
  cursorAccount: { repos: ["tuntharm/golfswing"], permissions: "pr-write" },
  onPullRequest: (_ctx, pr) => {
    if (pr.draft || !wakeOn.has(pr.action)) {
      return null;
    }
    return {
      auth: defaultGitHubAuth(_ctx),
      context: [
        "Look for a swing video in this pull request.",
        "Call analyze-swing on that file.",
        "The video it returns is the clubhead path at quarter speed. Shaft lines stay in the elements file and off the picture.",
      ],
    };
  },
  onComment: (ctx, comment) => defaultOnComment(ctx, comment, { botName: "golf-swing" }),
});