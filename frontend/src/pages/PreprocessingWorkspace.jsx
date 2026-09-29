import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  getLatestPreprocessing,
  preprocessingDownloadUrl,
  runPreprocessing,
} from "../services/preprocessingApi";
import "../styles/preprocessing.css";

function ProcessCard({ number, title, summary, children, open, onClick }) {
  return (
    <div className={`process-card ${open ? "expanded" : ""}`}>
      <button className="process-card-head" onClick={onClick}>
        <div className="process-number">{number}</div>
        <div className="process-title">
          <span>{title}</span>
          <small>{summary}</small>
        </div>
        <span className="expand-mark">{open ? "−" : "+"}</span>
      </button>
      {open && <div className="process-card-body">{children}</div>}
    </div>
  );
}

function DecisionCard({ decision, open, onClick }) {
  const explanation = decision.explanation || {};
  const observed = decision.observed || {};
  const missingPlan = decision.missing_plan || {};

  return (
    <div className={`decision-card ${open ? "expanded" : ""}`}>
      <button className="decision-head" onClick={onClick}>
        <div className="decision-main">
          <strong>{decision.column}</strong>
          <span>{decision.data_type || "Target / excluded"}</span>
        </div>
        <div className="decision-action">
          {decision.display_action || decision.action}
        </div>
        <span className="expand-mark">{open ? "−" : "+"}</span>
      </button>

      {open && (
        <div className="decision-body">
          <div className="decision-grid">
            <div>
              <span>Detected</span>
              <strong>{decision.data_type || "Target / excluded"}</strong>
            </div>
            <div>
              <span>Missing</span>
              <strong>
                {observed.missing_count ?? missingPlan.missing_count ?? 0}
                {" "}
                ({observed.missing_percent ?? missingPlan.missing_percent ?? 0}%)
              </strong>
            </div>
            <div>
              <span>Unique values</span>
              <strong>{observed.unique_count ?? decision.unique ?? "—"}</strong>
            </div>
            <div>
              <span>Status</span>
              <strong>
                {decision.action === "use" ? "✓ Automatically handled" : "Excluded"}
              </strong>
            </div>
          </div>

          <div className="explanation-block">
            <span className="explanation-label">WHY?</span>
            <p>{explanation.why || decision.reason || "No explanation recorded."}</p>
          </div>

          <div className="explanation-grid">
            <div>
              <span>What changed?</span>
              <p>{explanation.what_changed || "—"}</p>
            </div>
            <div>
              <span>Learned from</span>
              <p>{explanation.learned_from || "—"}</p>
            </div>
            <div>
              <span>Modeling effect</span>
              <p>{explanation.modeling_effect || "—"}</p>
            </div>
            <div>
              <span>Reason code</span>
              <p>{decision.reason_code || "—"}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function DownloadLink({ href, children }) {
  return (
    <a className="download-link" href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

export default function PreprocessingWorkspace() {
  const { datasetId } = useParams();
  const navigate = useNavigate();

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [openProcess, setOpenProcess] = useState(null);
  const [openDecision, setOpenDecision] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const runs = await getLatestPreprocessing(datasetId);
        if (runs.length) setResult(runs[0]);
      } catch {
        // A first-time user simply sees the start state.
      }
    }
    load();
  }, [datasetId]);

  async function run() {
    setLoading(true);
    setError("");

    try {
      const data = await runPreprocessing(datasetId, { mode: "automatic" });
      setResult(data);
      sessionStorage.setItem("automlPreprocessing", JSON.stringify(data));
      setOpenProcess(0);
    } catch (e) {
      setError(e.message || "Preprocessing failed.");
    } finally {
      setLoading(false);
    }
  }

  const report = result?.report;
  const summary = report?.summary || {};
  const decisions = report?.column_decisions || [];
  const outliers = report?.outliers || {};
  const split = report?.split || {};
  const transformation = report?.transformations || {};

  const download = (name) =>
    preprocessingDownloadUrl(datasetId, result.run_id, name);

  return (
    <div className="preprocess-page">
        {!result && !loading && (
          <section className="preprocess-start">
            <span className="section-label">MODULE 03 · PREPROCESSING</span>
            <h2>Prepare the dataset automatically</h2>
            <p>
              AutoML will clean modeling rows, decide how each usable feature
              should be represented, learn preprocessing from the training
              split, and prepare reproducible plain and scaled profiles.
            </p>

            <div className="workflow-note">
              <span>Automatic Mode</span>
              <strong>No decisions are requested from you during this run.</strong>
            </div>

            <button className="primary-button" onClick={run} disabled={loading}>
              Run Automatic Preprocessing
            </button>
          </section>
        )}

        {loading && (
          <div className="loading-state">
            <span className="spinner" />
            <div>
              <strong>Preparing your dataset...</strong>
              <span>
                Cleaning, splitting, learning preprocessing and recording decisions.
              </span>
            </div>
          </div>
        )}

        {error && <div className="message error-message">{error}</div>}

        {result && (
          <>
            <section className="preprocess-header-row">
              <div>
                <span className="section-label">MODULE 03 · COMPLETE</span>
                <h2>Automatic preprocessing finished</h2>
                <p className="header-description">
                  Every preprocessing decision is recorded so the next AutoML
                  stage can use the exact same fitted transformations.
                </p>
              </div>
              <span className="status-tag valid">Version {result.version}</span>
            </section>

            <section className="metric-grid">
              <div className="metric">
                <span>Rows</span>
                <strong>{summary.original_rows}</strong>
                <small>
                  {summary.after_cleaning} after modeling-row cleaning
                </small>
              </div>
              <div className="metric">
                <span>Features</span>
                <strong>{summary.used_features}</strong>
                <small>{summary.excluded_features} excluded</small>
              </div>
              <div className="metric">
                <span>Train / Test</span>
                <strong>{summary.train_rows} / {summary.test_rows}</strong>
                <small>{split.stratified ? "Stratified split" : "Random split"}</small>
              </div>
              <div className="metric">
                <span>Transformed</span>
                <strong>{summary.scaled_output_features}</strong>
                <small>features in scaled profile</small>
              </div>
            </section>

            <section className="workspace-section">
              <div className="section-heading">
                <div>
                  <span className="section-label">PROCESS</span>
                  <h2>What happened, in order</h2>
                </div>
              </div>

              <div className="process-stack">
                <ProcessCard
                  number="01"
                  title="Clean modeling rows"
                  summary={`${summary.missing_target_rows_removed || 0} missing-target + ${summary.duplicate_rows_removed || 0} duplicate rows removed`}
                  open={openProcess === 0}
                  onClick={() => setOpenProcess(openProcess === 0 ? null : 0)}
                >
                  <p>
                    Rows with a missing target cannot participate in supervised
                    learning, so they were removed from the derived modeling
                    dataset. Exact duplicate rows were also removed before the
                    train/test split. The validated source file itself was not changed.
                  </p>
                </ProcessCard>

                <ProcessCard
                  number="02"
                  title="Decide what each feature should do"
                  summary={`${summary.used_features} retained · ${summary.excluded_features} excluded`}
                  open={openProcess === 1}
                  onClick={() => setOpenProcess(openProcess === 1 ? null : 1)}
                >
                  <p>
                    Module 3 keeps the semantic decisions from Modules 1 and 2,
                    then applies its own missingness, cardinality and supported-type
                    rules. The detailed column cards below show the evidence for every decision.
                  </p>
                </ProcessCard>

                <ProcessCard
                  number="03"
                  title="Split before learning preprocessing"
                  summary={`${summary.train_rows} training rows · ${summary.test_rows} test rows`}
                  open={openProcess === 2}
                  onClick={() => setOpenProcess(openProcess === 2 ? null : 2)}
                >
                  <p>
                    The dataset is split using seed {split.random_seed}.{" "}
                    {split.stratified
                      ? "Because this is classification and the class counts permit it, stratification is used."
                      : "A non-stratified split was used because stratification was not feasible."}
                    {" "}No imputer, outlier boundary or scaler statistic is learned from the test set.
                  </p>
                </ProcessCard>

                <ProcessCard
                  number="04"
                  title="Learn missing values, outliers and encoding"
                  summary={`${summary.missing_values_handled} missing values handled · ${summary.outlier_values_capped_train} training outliers capped`}
                  open={openProcess === 3}
                  onClick={() => setOpenProcess(openProcess === 3 ? null : 3)}
                >
                  <div className="recipe-grid">
                    <div>
                      <strong>Missing values</strong>
                      <span>Numeric: median. Categorical: most-frequent below 5%, explicit missing category from 5%.</span>
                    </div>
                    <div>
                      <strong>Outliers</strong>
                      <span>Continuous numeric features use training-derived IQR capping. Rows are not deleted.</span>
                    </div>
                    <div>
                      <strong>Encoding</strong>
                      <span>Nominal and numeric-categorical values use one-hot encoding. Boolean values use 0/1.</span>
                    </div>
                    <div>
                      <strong>High cardinality</strong>
                      <span>Automatic V1 avoids target encoding and excludes unsafe high-cardinality features.</span>
                    </div>
                  </div>
                </ProcessCard>

                <ProcessCard
                  number="05"
                  title="Prepare scaling profiles"
                  summary="Plain + StandardScaler profiles"
                  open={openProcess === 4}
                  onClick={() => setOpenProcess(openProcess === 4 ? null : 4)}
                >
                  <p>
                    Two reproducible preprocessing profiles are prepared. The
                    <strong> plain </strong> profile does not scale values; the
                    <strong> scaled </strong> profile applies StandardScaler to
                    continuous numeric values. This keeps Module 3 model-agnostic:
                    downstream model selection can choose the profile required by a model.
                  </p>
                </ProcessCard>

                <ProcessCard
                  number="06"
                  title="Record and preserve everything"
                  summary={`Run ${result.run_id}`}
                  open={openProcess === 5}
                  onClick={() => setOpenProcess(openProcess === 5 ? null : 5)}
                >
                  <p>
                    The validated dataset fingerprint, split row IDs, fitted
                    preprocessors, feature mapping and decision report are stored
                    with this run. Re-running the same version uses a deterministic
                    seed and never edits the original validated dataset.
                  </p>
                </ProcessCard>
              </div>
            </section>

            <section className="workspace-section">
              <div className="section-heading">
                <div>
                  <span className="section-label">COLUMN DECISIONS</span>
                  <h2>Automatic decisions</h2>
                  <p>Click any card to expand the complete dataset-specific explanation.</p>
                </div>
              </div>

              <div className="decision-stack">
                {decisions.map((decision, index) => (
                  <DecisionCard
                    key={`${decision.column}-${index}`}
                    decision={decision}
                    open={openDecision === index}
                    onClick={() =>
                      setOpenDecision(openDecision === index ? null : index)
                    }
                  />
                ))}
              </div>
            </section>

            <section className="workspace-section">
              <div className="section-heading">
                <div>
                  <span className="section-label">PREPROCESSING RECIPE</span>
                  <h2>What the next stage receives</h2>
                </div>
              </div>

              <div className="recipe-grid large">
                <div>
                  <strong>Missing values</strong>
                  <span>
                    {transformation.missing_values?.numeric}; categorical values
                    use the recorded missingness threshold strategy.
                  </span>
                </div>
                <div>
                  <strong>Encoding</strong>
                  <span>{transformation.encoding?.nominal_categorical}</span>
                </div>
                <div>
                  <strong>Outliers</strong>
                  <span>
                    {transformation.outliers?.method} · training split only · no row deletion
                  </span>
                </div>
                <div>
                  <strong>Scaling</strong>
                  <span>{transformation.scaling?.scaled_profile}</span>
                </div>
              </div>
            </section>

            {Object.keys(outliers).length > 0 && (
              <section className="workspace-section">
                <div className="section-heading">
                  <div>
                    <span className="section-label">OUTLIER DETAILS</span>
                    <h2>Training-derived IQR treatment</h2>
                  </div>
                </div>
                <div className="outlier-table">
                  {Object.entries(outliers).map(([column, value]) => (
                    <div className="outlier-row" key={column}>
                      <strong>{column}</strong>
                      <span>
                        Bounds {Number(value.lower).toFixed(3)} to {Number(value.upper).toFixed(3)}
                      </span>
                      <span>{value.train_values_capped} train capped</span>
                      <span>{value.test_values_capped} test capped</span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {(report?.warnings || []).length > 0 && (
              <section className="workspace-section">
                <div className="section-heading">
                  <div>
                    <span className="section-label">WARNINGS</span>
                    <h2>Recorded observations</h2>
                  </div>
                </div>
                <div className="issue-list">
                  {report.warnings.map((warning, index) => (
                    <div className="issue-row" key={index}>
                      <span className="issue-marker">!</span>
                      <div>
                        <strong>{warning.code}</strong>
                        <p>{warning.message}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section className="workspace-section">
              <div className="section-heading">
                <div>
                  <span className="section-label">ARTIFACTS</span>
                  <h2>Module 3 outputs</h2>
                  <p>Downloads stay inside the workspace; there is no separate Downloads page.</p>
                </div>
              </div>

              <div className="download-grid">
                <DownloadLink href={download("report")}>Preprocessing report</DownloadLink>
                <DownloadLink href={download("config")}>Preprocessing configuration</DownloadLink>
                <DownloadLink href={download("feature_mapping")}>Feature mapping</DownloadLink>
                <DownloadLink href={download("processed_train_plain")}>Processed train · plain</DownloadLink>
                <DownloadLink href={download("processed_test_plain")}>Processed test · plain</DownloadLink>
                <DownloadLink href={download("processed_train_scaled")}>Processed train · scaled</DownloadLink>
                <DownloadLink href={download("processed_test_scaled")}>Processed test · scaled</DownloadLink>
              </div>
            </section>

            <div className="preprocess-actions">
              <button
                className="secondary-button"
                onClick={() => navigate(`/datasets/${datasetId}`)}
              >
                Back to Dataset
              </button>
              <button
                className="primary-button"
                onClick={() => navigate(`/datasets/${datasetId}/model-selection`)}
              >
                Continue to Model Selection
              </button>
            </div>
          </>
        )}
    </div>
  );
}
