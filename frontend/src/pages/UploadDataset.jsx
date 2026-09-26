import {
  useRef,
  useState
} from "react";

import {
  useNavigate
} from "react-router-dom";

import {
  uploadDataset
} from "../services/datasetApi";


function UploadDataset() {

  const navigate = useNavigate();

  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);

  const [isUploading, setIsUploading] = useState(false);

  const [progress, setProgress] = useState(0);

  const [error, setError] = useState("");

  const [isDragging, setIsDragging] = useState(false);


  function handleFileSelection(file) {

    setError("");

    if (!file) {
      return;
    }

    const extension = file.name
      .split(".")
      .pop()
      .toLowerCase();

    if (
      extension !== "csv" &&
      extension !== "xlsx"
    ) {

      setSelectedFile(null);

      setError(
        "Only CSV and XLSX files are supported."
      );

      return;
    }

    if (file.size === 0) {

      setSelectedFile(null);

      setError(
        "The selected file is empty."
      );

      return;
    }

    setSelectedFile(file);
  }


  function handleInputChange(event) {

    const file = event.target.files?.[0];

    handleFileSelection(file);
  }


  function handleDrop(event) {

    event.preventDefault();

    setIsDragging(false);

    const file = event.dataTransfer.files?.[0];

    handleFileSelection(file);
  }


  async function handleUpload() {

    if (!selectedFile) {

      setError(
        "Please select a dataset first."
      );

      return;
    }

    setIsUploading(true);

    setError("");

    setProgress(0);

    try {

      const result = await uploadDataset(
        selectedFile,
        (event) => {

          if (event.total) {

            const percentage = Math.round(
              (event.loaded / event.total) * 100
            );

            setProgress(percentage);
          }
        }
      );

      navigate(
        `/analysis/${result.dataset_id}`
      );

    } catch (err) {

      const backendMessage =
        err.response?.data?.detail;

      if (
        typeof backendMessage === "string"
      ) {

        setError(
          backendMessage
        );

      } else if (
        backendMessage?.message
      ) {

        setError(
          backendMessage.message
        );

      } else {

        setError(
          "Dataset upload failed. Please try again."
        );
      }

    } finally {

      setIsUploading(false);
    }
  }


  return (
    <div className="app-shell">

      <header className="topbar">

        <div className="brand">

          <div className="brand-mark">
            A
          </div>

          <div>
            <h1>AutoML Studio</h1>

            <p>
              Intelligent machine learning pipeline
            </p>
          </div>

        </div>

      </header>


      <main className="page-container">

        <section className="page-heading">

          <span className="eyebrow">
            MODULE 1
          </span>

          <h2>
            Upload Dataset
          </h2>

          <p>
            Upload your tabular dataset and
            let AutoML Studio inspect its
            structure and quality.
          </p>

        </section>


        <section className="upload-card">

          <div
            className={
              `drop-zone ${
                isDragging
                  ? "drop-zone-active"
                  : ""
              }`
            }

            onDragOver={(event) => {

              event.preventDefault();

              setIsDragging(true);
            }}

            onDragLeave={() => {
              setIsDragging(false);
            }}

            onDrop={handleDrop}

            onClick={() => {
              fileInputRef.current?.click();
            }}
          >

            <div className="upload-icon">
              ↑
            </div>

            <h3>
              Drop your dataset here
            </h3>

            <p>
              or click to browse your files
            </p>

            <span className="file-types">
              Supported formats: CSV, XLSX
            </span>

            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx"
              onChange={handleInputChange}
              hidden
            />

          </div>


          {selectedFile && (

            <div className="selected-file">

              <div>

                <strong>
                  {selectedFile.name}
                </strong>

                <span>
                  {(
                    selectedFile.size /
                    1024
                  ).toFixed(1)} KB
                </span>

              </div>

              <button
                className="remove-button"
                onClick={() => {
                  setSelectedFile(null);
                  setError("");
                }}
              >
                Remove
              </button>

            </div>
          )}


          {error && (

            <div className="error-box">
              {error}
            </div>
          )}


          {isUploading && (

            <div className="progress-section">

              <div className="progress-header">

                <span>
                  Uploading and analyzing dataset...
                </span>

                <strong>
                  {progress}%
                </strong>

              </div>

              <div className="progress-track">

                <div
                  className="progress-fill"
                  style={{
                    width: `${progress}%`
                  }}
                />

              </div>

              <p>
                AutoML is loading, validating
                and profiling your dataset.
              </p>

            </div>
          )}


          <button
            className="primary-button upload-button"
            disabled={
              !selectedFile ||
              isUploading
            }
            onClick={handleUpload}
          >
            {isUploading
              ? "Analyzing Dataset..."
              : "Upload & Analyze"}
          </button>

        </section>


        <section className="info-grid">

          <div className="info-card">

            <div className="info-number">
              01
            </div>

            <h3>
              Validate
            </h3>

            <p>
              File format, encoding,
              delimiter and structural
              checks are performed.
            </p>

          </div>


          <div className="info-card">

            <div className="info-number">
              02
            </div>

            <h3>
              Inspect
            </h3>

            <p>
              Missing values, duplicates,
              data types and ID candidates
              are identified.
            </p>

          </div>


          <div className="info-card">

            <div className="info-number">
              03
            </div>

            <h3>
              Report
            </h3>

            <p>
              A unified dataset Quality
              Report is generated for
              the next pipeline stage.
            </p>

          </div>

        </section>

      </main>

    </div>
  );
}


export default UploadDataset;