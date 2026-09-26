import {
  useEffect,
  useState
} from "react";

import {
  useNavigate,
  useParams
} from "react-router-dom";

import {
  validateTarget
} from "../services/targetApi";

import "../styles/targetValidation.css";


function TargetValidation() {

  const navigate = useNavigate();

  const { datasetId: routeDatasetId } =
    useParams();


  // ========================================================
  // STATE
  // ========================================================

  const [
    dataset,
    setDataset
  ] = useState(null);


  const [
    selectedTarget,
    setSelectedTarget
  ] = useState("");


  const [
    result,
    setResult
  ] = useState(null);


  const [
    loading,
    setLoading
  ] = useState(false);


  const [
    error,
    setError
  ] = useState("");


  // ========================================================
  // LOAD DATASET
  // ========================================================

  useEffect(() => {

    const storedDataset =
      sessionStorage.getItem(
        "automlDataset"
      );


    if (!storedDataset) {

      setError(
        "Dataset information is missing."
      );

      return;

    }


    try {

      const parsed =
        JSON.parse(
          storedDataset
        );


      const loadedDataset =
        parsed.dataset || parsed;


      setDataset(
        loadedDataset
      );


      /*
       * If DatasetAnalysis already stored
       * a selected target, show it here.
       */

      const storedTarget =
        sessionStorage.getItem(
          "automlSelectedTarget"
        );


      if (storedTarget) {

        setSelectedTarget(
          storedTarget
        );

      }

    }

    catch {

      setError(
        "Unable to read dataset information."
      );

    }

  }, []);


  // ========================================================
  // DATASET INFORMATION
  // ========================================================

  const datasetId =
    routeDatasetId ||
    dataset?.dataset_id;


  const columns =
    dataset?.column_names || [];


  // ========================================================
  // VALIDATE TARGET
  // ========================================================

  async function handleValidate() {

    if (!selectedTarget) {

      setError(
        "Please select a target column."
      );

      return;

    }


    if (!datasetId) {

      setError(
        "Dataset ID is missing."
      );

      return;

    }


    setError("");

    setResult(null);

    setLoading(true);


    try {

      console.log(
        "Checking target:",
        selectedTarget
      );

      console.log(
        "Dataset ID:",
        datasetId
      );


      const validationResult =
        await validateTarget(
          datasetId,
          selectedTarget
        );


      console.log(
        "Validation result:",
        validationResult
      );


      setResult(
        validationResult
      );


      sessionStorage.setItem(
        "automlTarget",
        JSON.stringify(
          validationResult
        )
      );

    }

    catch (validationError) {

      console.error(
        "Target validation error:",
        validationError
      );


      setError(
        validationError.message ||
        "Unable to validate target."
      );

    }

    finally {

      setLoading(false);

    }

  }


  // ========================================================
  // CONTINUE
  // ========================================================

  function handleContinue() {

    if (!result) {
      return;
    }


    if (
      result.status ===
      "invalid"
    ) {

      return;

    }


    /*
     * Module 3 / Problem Definition
     * has not been implemented yet.
     *
     * For now, we simply keep the validated
     * result on this page.
     */

    console.log(
      "Target validated successfully:",
      result
    );

  }


  // ========================================================
  // STATUS TITLE
  // ========================================================

  function getStatusTitle() {

    if (!result) {
      return "";
    }


    if (
      result.status ===
      "valid"
    ) {

      return "Valid Target";

    }


    if (
      result.status ===
      "valid_with_warnings"
    ) {

      return "Valid Target with Warnings";

    }


    return "Cannot Be Used as Target";

  }


  // ========================================================
  // LOADING
  // ========================================================

  if (!dataset && !error) {

    return (
      <div className="target-page">

        <div className="target-loading">

          Loading dataset...

        </div>

      </div>
    );

  }


  // ========================================================
  // PAGE
  // ========================================================

  return (

    <div className="target-page">

      <div className="target-container">


        {/* HEADER */}

        <div className="target-header">

          <span className="module-label">
            MODULE 2
          </span>

          <h1>
            Target Validation
          </h1>

          <p>
            Choose the column you want
            AutoML to predict. The system
            will validate your choice.
          </p>

        </div>


        {/* DATASET SUMMARY */}

        {dataset && (

          <div className="dataset-summary">

            <div>

              <span>
                Dataset
              </span>

              <strong>
                {dataset.original_filename}
              </strong>

            </div>


            <div>

              <span>
                Rows
              </span>

              <strong>
                {dataset.rows}
              </strong>

            </div>


            <div>

              <span>
                Columns
              </span>

              <strong>
                {dataset.columns}
              </strong>

            </div>

          </div>

        )}


        {/* TARGET SELECTION */}

        <div className="target-card">

          <label htmlFor="target-column">

            Target Column

          </label>


          <p className="target-help">

            Select the outcome you want the
            model to predict. AutoML will
            not automatically choose a target
            for you.

          </p>


          <select
            id="target-column"
            value={selectedTarget}
            onChange={(event) => {

              setSelectedTarget(
                event.target.value
              );

              setResult(null);

              setError("");

            }}
          >

            <option value="">
              Select a column
            </option>


            {columns.map(
              (column) => (

                <option
                  key={column}
                  value={column}
                >
                  {column}
                </option>

              )
            )}

          </select>


          <button
            className="validate-button"
            onClick={handleValidate}
            disabled={
              !selectedTarget ||
              loading
            }
          >

            {loading
              ? "Checking..."
              : "Check Target"
            }

          </button>

        </div>


        {/* ERROR */}

        {error && (

          <div className="target-error">

            <strong>
              Error
            </strong>

            <span>
              {error}
            </span>

          </div>

        )}


        {/* RESULT */}

        {result && (

          <div
            className={
              `target-result ${result.status}`
            }
          >

            {/* RESULT HEADER */}

            <div className="result-heading">

              <div>

                <span className="result-label">
                  Target Analysis
                </span>

                <h2>
                  {selectedTarget}
                </h2>

              </div>


              <span className="status-badge">

                {getStatusTitle()}

              </span>

            </div>


            {/* BASIC DETAILS */}

            <div className="analysis-grid">

              <div className="analysis-item">

                <span>
                  Data Type
                </span>

                <strong>
                  {result.target_data_type ||
                    "—"}
                </strong>

              </div>


              <div className="analysis-item">

                <span>
                  Semantic Role
                </span>

                <strong>
                  {result.target_semantic_role ||
                    "—"}
                </strong>

              </div>


              <div className="analysis-item">

                <span>
                  Valid Values
                </span>

                <strong>
                  {
                    result.target_statistics
                      ?.valid_row_count ??
                    "—"
                  }
                </strong>

              </div>


              <div className="analysis-item">

                <span>
                  Missing
                </span>

                <strong>

                  {
                    result.target_statistics
                      ?.missing_percentage ??
                    "—"
                  }%

                </strong>

              </div>


              <div className="analysis-item">

                <span>
                  Unique Values
                </span>

                <strong>

                  {
                    result.target_statistics
                      ?.unique_count ??
                    "—"
                  }

                </strong>

              </div>

            </div>


            {/* PROBLEM TYPE */}

            {result.problem_type && (

              <div className="problem-box">

                <span>
                  Problem Type
                </span>

                <strong>

                  {result.sub_problem_type
                    ? result.sub_problem_type
                        .replace(
                          /_/g,
                          " "
                        )
                    : ""}

                  {" — "}

                  {result.problem_type}

                </strong>

              </div>

            )}


            {/* CLASS DISTRIBUTION */}

            {
              result.class_distribution
                ?.classes &&
              (

                <div className="distribution-section">

                  <h3>
                    Class Distribution
                  </h3>


                  <div className="class-list">

                    {Object.entries(
                      result.class_distribution
                        .classes
                    ).map(
                      (
                        [
                          className,
                          classData
                        ]
                      ) => (

                        <div
                          className="class-row"
                          key={className}
                        >

                          <span>
                            {className}
                          </span>

                          <strong>
                            {classData.count}
                          </strong>

                          <span>
                            {classData.percentage}%
                          </span>

                        </div>

                      )
                    )}

                  </div>


                  {
                    result.class_distribution
                      .imbalance_ratio !==
                      undefined &&
                    (

                      <p className="distribution-info">

                        Imbalance ratio:
                        {" "}
                        {
                          result.class_distribution
                            .imbalance_ratio
                        }:1

                      </p>

                    )
                  }

                </div>

              )
            }


            {/* ERRORS */}

            {
              result.errors?.length > 0 &&
              (

                <div className="issues error-issues">

                  <h3>
                    Why this cannot be a target
                  </h3>


                  {result.errors.map(
                    (
                      issue,
                      index
                    ) => (

                      <div
                        className="issue"
                        key={index}
                      >

                        <strong>
                          {issue.code}
                        </strong>

                        <span>
                          {issue.message}
                        </span>

                      </div>

                    )
                  )}

                </div>

              )
            }


            {/* WARNINGS */}

            {
              result.warnings?.length > 0 &&
              (

                <div className="issues warning-issues">

                  <h3>
                    Warnings
                  </h3>


                  {result.warnings.map(
                    (
                      issue,
                      index
                    ) => (

                      <div
                        className="issue"
                        key={index}
                      >

                        <strong>
                          {issue.code}
                        </strong>

                        <span>
                          {issue.message}
                        </span>

                      </div>

                    )
                  )}

                </div>

              )
            }


            {/* FEATURES */}

            {
              result.status !== "invalid" &&
              (

                <div className="features-section">

                  <h3>
                    Feature Availability
                  </h3>


                  <p>

                    {
                      result.usable_feature_count ??
                      0
                    }

                    {" "}
                    usable feature(s)

                  </p>


                  {
                    result.usable_features
                      ?.length > 0 &&
                    (

                      <div className="feature-list">

                        {
                          result.usable_features
                            .map(
                              (feature) => (

                                <span
                                  key={feature}
                                  className="feature-pill"
                                >
                                  {feature}
                                </span>

                              )
                            )
                        }

                      </div>

                    )
                  }

                </div>

              )
            }


            {/* ACTION */}

            <div className="result-actions">

              {result.status === "invalid" ? (

                <p className="invalid-message">

                  Please select another target
                  column.

                </p>

              ) : (

                <button
                  className="continue-button"
                  onClick={
                    handleContinue
                  }
                >

                  Continue

                </button>

              )}

            </div>

          </div>

        )}

      </div>

    </div>

  );

}


export default TargetValidation;