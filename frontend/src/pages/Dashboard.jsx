import { useNavigate } from "react-router-dom";
import AppShell from "../components/AppShell";

function Dashboard() {
  const navigate = useNavigate();

  return (
    <AppShell
      title="Build a model without the guesswork."
      subtitle="Upload a dataset, choose what you want to predict, and let AutoML handle the machine-learning pipeline."
    >
      <section className="dashboard-hero">
        <div>
          <span className="eyebrow">AUTOML WORKSPACE</span>
          <h2>Start with your dataset.</h2>
          <p>
            The first run is always automatic. AutoML analyzes the data,
            prepares it, selects models, tunes them, evaluates the results,
            and explains the final pipeline.
          </p>
          <button className="primary-button" onClick={() => navigate("/upload")}>
            Upload Dataset
          </button>
        </div>
        <div className="hero-flow" aria-label="Automatic workflow">
          <div className="flow-row"><span>01</span><strong>Upload</strong><small>CSV or XLSX</small></div>
          <div className="flow-line" />
          <div className="flow-row"><span>02</span><strong>Define target</strong><small>You choose what to predict</small></div>
          <div className="flow-line" />
          <div className="flow-row"><span>03</span><strong>Run AutoML</strong><small>Automatic first run</small></div>
          <div className="flow-line" />
          <div className="flow-row"><span>04</span><strong>Review results</strong><small>Inspect or customize</small></div>
        </div>
      </section>

      <section className="dashboard-grid">
        <article className="plain-panel">
          <div className="panel-heading">
            <div>
              <span className="section-label">CURRENT WORKFLOW</span>
              <h3>One workspace, not a chain of pages.</h3>
            </div>
          </div>
          <div className="workflow-list">
            <div><span>01</span><strong>Dataset analysis</strong><p>Structure, quality, types, missing values and identifiers.</p></div>
            <div><span>02</span><strong>Target validation</strong><p>You select the target; AutoML determines whether it is usable and how to model it.</p></div>
            <div><span>03</span><strong>Automatic pipeline</strong><p>Preprocessing through evaluation runs as one pipeline.</p></div>
            <div><span>04</span><strong>Guided customization</strong><p>Only after the first run can you edit stages and create another version.</p></div>
          </div>
        </article>

        <article className="plain-panel guest-panel">
          <span className="section-label">GUEST WORKSPACE</span>
          <h3>Try the workflow first.</h3>
          <p>
            Guest sessions can upload, analyze, run AutoML, and download outputs.
            Persistent datasets and previous runs will be available after authentication is connected.
          </p>
          <div className="guest-note">
            <span className="status-dot" />
            Nothing is added to a personal dataset library while you are a guest.
          </div>
        </article>
      </section>
    </AppShell>
  );
}

export default Dashboard;
