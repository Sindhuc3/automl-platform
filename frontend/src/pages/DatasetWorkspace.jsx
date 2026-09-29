import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import AppShell from "../components/AppShell";
import { getDataset } from "../services/datasetApi";
import { validateTarget } from "../services/targetApi";

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function formatType(value) {
  return String(value || "unknown").replaceAll("_", " ");
}

function statusLabel(status) {
  if (status === "clean") return "Clean";
  if (status === "usable_with_warnings") return "Usable with warnings";
  return "Blocked";
}

function DatasetWorkspace() {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const [dataset, setDataset] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [targetResult, setTargetResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadDataset() {
      try {
        setLoading(true);
        const result = await getDataset(datasetId);
        setDataset(result);
        if (result.target_column) setTargetColumn(result.target_column);
        sessionStorage.setItem("automlDataset", JSON.stringify(result));
      } catch (err) {
        setError(err.response?.data?.detail || "Unable to load dataset.");
      } finally {
        setLoading(false);
      }
    }

    loadDataset();
  }, [datasetId]);

  const qualityReport = dataset?.quality_report;
  const profile = qualityReport?.profile;
  const columnTypes = qualityReport?.column_types || {};
  const idCandidates = qualityReport?.id_candidates || [];
  const idColumnNames = useMemo(
    () => new Set(idCandidates.map((item) => item.column)),
    [idCandidates]
  );

  async function handleValidateTarget() {
    if (!targetColumn) {
      setError("Please select a target column.");
      return;
    }

    setError("");
    setTargetResult(null);
    setValidating(true);

    try {
      const result = await validateTarget(datasetId, targetColumn);
      setTargetResult(result);
      sessionStorage.setItem("automlSelectedTarget", targetColumn);
      sessionStorage.setItem("automlTarget", JSON.stringify(result));
    } catch (err) {
      setError(err.message || "Unable to validate target.");
    } finally {
      setValidating(false);
    }
  }

  if (loading) {
    return (
      <AppShell title="Dataset Workspace" subtitle="Loading dataset analysis...">
        <div className="loading-state"><span className="spinner" />Loading dataset...</div>
      </AppShell>
    );
  }

  if (error && !dataset) {
    return (
      <AppShell title="Dataset Workspace" subtitle="The dataset could not be loaded.">
        <div className="message error-message">{error}</div>
        <button className="secondary-button" onClick={() => navigate("/upload")}>Back to upload</button>
      </AppShell>
    );
  }

  return (
    <AppShell
      title="Dataset Analysis"
      subtitle="Review the dataset quality, inspect detected columns, and choose the target for the first automatic run."
      datasetName={dataset.original_filename}
    >
      <div className="workspace-toolbar">
        <button className="text-button" onClick={() => navigate("/upload")}>← New dataset</button>
        <span>Dataset ID: <strong>{dataset.dataset_id}</strong></span>
      </div>

      <section className="quality-strip">
        <div>
          <span className="section-label">DATASET QUALITY</span>
          <strong>{statusLabel(qualityReport?.status)}</strong>
        </div>
        <div className="quality-stat"><strong>{qualityReport?.summary?.errors ?? 0}</strong><span>Errors</span></div>
        <div className="quality-stat"><strong>{qualityReport?.summary?.warnings ?? 0}</strong><span>Warnings</span></div>
        <div className="quality-stat"><strong>{qualityReport?.summary?.information ?? 0}</strong><span>Information</span></div>
      </section>

      <section className="metric-grid">
        <div className="metric"><span>Rows</span><strong>{formatNumber(profile?.rows)}</strong></div>
        <div className="metric"><span>Columns</span><strong>{formatNumber(profile?.columns)}</strong></div>
        <div className="metric"><span>Memory</span><strong>{profile?.memory?.mb ?? 0} MB</strong></div>
        <div className="metric"><span>Duplicate rows</span><strong>{formatNumber(profile?.duplicate_rows?.count)}</strong></div>
      </section>

      {qualityReport?.warnings?.length > 0 && (
        <section className="workspace-section">
          <div className="section-heading"><div><span className="section-label">QUALITY REPORT</span><h2>Important observations</h2></div></div>
          <div className="issue-list">
            {qualityReport.warnings.slice(0, 5).map((warning, index) => (
              <div className="issue-row" key={`${warning.code}-${index}`}>
                <span className="issue-marker">!</span>
                <div><strong>{formatType(warning.code)}</strong><p>{warning.message}</p></div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="workspace-section">
        <div className="section-heading">
          <div><span className="section-label">COLUMN ANALYSIS</span><h2>Detected structure</h2></div>
          <span className="section-meta">{profile?.columns || 0} columns</span>
        </div>
        <div className="table-shell">
          <table>
            <thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Role signal</th></tr></thead>
            <tbody>
              {(profile?.column_names || []).map((column) => {
                const missing = profile?.missing_values?.[column] || {};
                const unique = profile?.unique_values?.[column] ?? 0;
                const isId = idColumnNames.has(column);
                return (
                  <tr key={column}>
                    <td><strong>{column}</strong></td>
                    <td><span className="type-tag">{formatType(columnTypes[column])}</span></td>
                    <td>{formatNumber(missing.count)}</td>
                    <td>{formatNumber(unique)}</td>
                    <td>{isId ? <span className="signal-tag">Identifier candidate</span> : <span className="muted">—</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="target-section">
        <div className="target-copy">
          <span className="section-label">TARGET</span>
          <h2>What do you want the model to predict?</h2>
          <p>You choose the target. AutoML only validates whether that choice is suitable and determines whether the task is classification or regression.</p>
        </div>
        <div className="target-controls">
          <label htmlFor="target-column">Target column</label>
          <select id="target-column" value={targetColumn} onChange={(event) => { setTargetColumn(event.target.value); setTargetResult(null); setError(""); }}>
            <option value="">Select a column</option>
            {(dataset.column_names || []).map((column) => <option key={column} value={column}>{column}</option>)}
          </select>
          <button className="primary-button" disabled={!targetColumn || validating} onClick={handleValidateTarget}>
            {validating ? "Checking target..." : "Check Target"}
          </button>
        </div>
      </section>

      {error && <div className="message error-message">{error}</div>}

      {targetResult && (
        <section className={`target-result-panel ${targetResult.status === "invalid" ? "invalid" : "valid"}`}>
          <div className="result-topline">
            <div>
              <span className="section-label">TARGET VALIDATION</span>
              <h2>{targetResult.status === "invalid" ? "Target cannot be used" : "Target is ready"}</h2>
            </div>
            <span className={`status-tag ${targetResult.status}`}>{formatType(targetResult.status)}</span>
          </div>

          {targetResult.status !== "invalid" && (
            <>
              <div className="problem-definition">
                <div><span>Target</span><strong>{targetResult.target_column || targetColumn}</strong></div>
                <div><span>Problem type</span><strong>{formatType(targetResult.problem_type)}</strong></div>
                {targetResult.sub_problem_type && <div><span>Task</span><strong>{formatType(targetResult.sub_problem_type)}</strong></div>}
                <div><span>Valid rows</span><strong>{formatNumber(targetResult.target_statistics?.valid_row_count)}</strong></div>
                <div><span>Missing</span><strong>{targetResult.target_statistics?.missing_percentage ?? 0}%</strong></div>
                <div><span>Usable features</span><strong>{formatNumber(targetResult.usable_feature_count)}</strong></div>
              </div>

              {targetResult.warnings?.length > 0 && (
                <div className="compact-issues">
                  {targetResult.warnings.map((warning, index) => <div key={`${warning.code}-${index}`}><strong>{formatType(warning.code)}</strong><span>{warning.message}</span></div>)}
                </div>
              )}

              <div className="next-stage-note">
                <strong>Ready for the automatic pipeline.</strong>
                <span>Pipeline execution will be connected as the remaining ML modules are added.</span>
              </div>
            </>
          )}

          {targetResult.status === "invalid" && (
            <div className="compact-issues error-list">
              {(targetResult.errors || []).map((issue, index) => <div key={`${issue.code}-${index}`}><strong>{formatType(issue.code)}</strong><span>{issue.message}</span></div>)}
            </div>
          )}
        </section>
      )}
    </AppShell>
  );
}

export default DatasetWorkspace;
