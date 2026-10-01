# Module 6.5 — Selection Arbiter

M6.5 converts M6 HPO evidence into one deterministic `FinalModelSpec` for M7.

## Selection policy

1. Primary metric is fixed by M5 before candidate results are used.
2. Reject candidates that fail required guardrails.
3. Require evidence that the candidate beats the M5 Dummy baseline when that evidence is present.
4. Identify the best CV score using the correct metric direction.
5. Build the one-corrected-SE tie zone around the best candidate.
6. Within that zone, prefer lower model complexity, then fewer features, then lower fit cost, then deterministic IDs.
7. Emit finalists plus the selected `FinalModelSpec` for M7.

The sealed test set is never read by M6.5.

## Output

`m7_handoff.final_model_spec` contains:

- algorithm_id
- feature_set_id
- task
- primary_metric
- metric_direction
- tuned hyperparameters
- CV score
- corrected SE
- feature count when available
- selection evidence

## Tests

`13 passed` including four M6.5 arbiter tests.
