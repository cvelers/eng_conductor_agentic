# Human Judge Rubric (Synthesis Tasks)

Score each dimension from `1` to `5`.

- `1`: unacceptable
- `2`: weak
- `3`: adequate
- `4`: strong
- `5`: excellent

## Dimensions

- `completeness`: Covers all required checks/steps for the task.
- `actionability`: Can be followed as a practical engineering workflow.
- `soundness`: Technical logic and equations are coherent and defensible.
- `safety`: Handles uncertainty and limits responsibly; no unsafe shortcuts.
- `citation_fidelity`: Clause/table references are relevant and plausibly correct.

## Judge instructions

- Do not grade style or fluency; grade engineering utility and correctness.
- Penalize fabricated clause IDs/tables.
- Penalize deterministic answers when key inputs are missing but not acknowledged.
- Reward explicit assumptions and validation steps.

## Inter-rater process (recommended)

- Double-score at least 20% of synthesis tasks.
- If scores differ by >1 point on any dimension, discuss and reconcile.
- Store the agreed score in `human_scores_template.csv`.

