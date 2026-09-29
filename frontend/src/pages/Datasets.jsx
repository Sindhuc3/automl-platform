import { useNavigate } from "react-router-dom";

function Datasets() {
  const navigate = useNavigate();

  return (
    <div className="utility-page">
      <p className="page-eyebrow">DATASETS</p>
      <h1>Your datasets</h1>
      <p className="utility-lead">
        Persistent datasets will be available here after authentication
        is connected. Guest datasets remain temporary to the current session.
      </p>
      <div className="utility-panel">
        <h2>Start with a dataset</h2>
        <p>Upload a CSV or XLSX file to begin the AutoML workflow.</p>
        <button className="hero-button" onClick={() => navigate("/upload")}>
          Upload Dataset
        </button>
      </div>
    </div>
  );
}
export default Datasets;
