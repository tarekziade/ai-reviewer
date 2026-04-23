Review this repository as infrastructure code for a GitHub-native AI code reviewer.

Prioritize:
- Broken GitHub API usage, wrong endpoints, missing permission scopes, and event payload mismatches.
- Diff parsing, line anchoring, and review-comment validation bugs that would attach comments to the wrong place or drop valid comments.
- Trigger gating and trust-boundary problems around `issue_comment`, `pull_request_review_comment`, author association checks, PR state checks, and default-branch rule loading.
- Prompt-safety regressions, malformed JSON handling, and cases where model output can crash or derail the review flow.
- Secret handling and logging mistakes that could expose tokens or sensitive request data.
- Divergence between Action mode and App mode when a change silently fixes one path and breaks the other.

Deprioritize:
- Style-only feedback, formatting nits, and speculative refactors.
- Requests for extra abstractions unless the existing code is already causing a concrete correctness or maintenance problem.

Prefer concise comments tied to observable runtime behavior.
