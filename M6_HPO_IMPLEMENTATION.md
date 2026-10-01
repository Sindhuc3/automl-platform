# Module 6 — Hyperparameter Optimization (Automatic Mode)

## Purpose
M6 tunes only the candidates explicitly promoted by Module 5. It never widens the M5 handoff and never reads the sealed test set.

## Flow
```text
M5 Screening
  -> eligible_for_hpo (<= 4)
  -> M6 HPO
       trial 0 = algorithm default configuration
       trials 1..N = deterministic samples from algorithm-specific search space
       same frozen M5 CV folds
       same feature-set recipe inside each fold
       bounded time/fit budgets
  -> tuned candidate evidence
  -> M6.5 Selection Arbiter
```

## Default budgets
- 25 trials per promoted candidate
- at most 4 promoted candidates
- 600 s global budget
- 180 s per candidate
- 600 fits
- deterministic seed 42
- no test-set access

## API
- `POST /api/datasets/{dataset_id}/model-hpo`
  - optional `screening_id`
  - optional `config`
- `GET /api/datasets/{dataset_id}/model-hpo`
- `GET /api/datasets/{dataset_id}/model-hpo/{hpo_id}`

## Important contract
M6 requires every promoted M5 candidate to have:
- `guardrail_pass == true`
- `baseline_evidence.beats_baseline == true`
- `within_one_se_of_best == true`

If M5 sends more than four candidates, M6 rejects the handoff instead of silently widening or truncating it.

## Reproducibility
M5 profile outputs persist their materialized CV plan. M6 reconstructs that exact plan and reuses the same fold indices. HPO execution IDs are unique per run; a separate fingerprint is retained for reproducibility.
