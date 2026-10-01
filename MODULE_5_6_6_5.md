# Modules 5, 6 and 6.5

## M5 — Model Screening
- Frozen, reusable CV plan.
- Dummy baseline.
- Algorithm × feature-set screening.
- Primary metric is fixed before candidate results.
- Paired baseline evidence and corrected uncertainty.
- Guardrails and bounded execution.

## M6 — Hyperparameter Optimization
- Consumes M5 successful candidates.
- Uses bounded deterministic randomized search over the algorithm registry search spaces.
- Reuses the exact M5 CV folds.
- Evaluates feature-set recipes inside each fold.
- Default budget: 25 trials/candidate, max 3 promoted candidates, bounded global fits/time.
- Produces tuned candidates, best parameters, corrected SE, baseline evidence, and M7/M6.5 handoff.
- Never uses the sealed test set.

## M6.5 — Selection Arbiter
- Consumes M6 CV-only evidence.
- Applies guardrails, baseline evidence, primary metric direction, 1-SE tie zone, complexity, feature count and fit cost.
- Produces exactly one `final_model_spec` for M7 when a candidate passes.
- Explicitly reports `test_set_used=False`.
