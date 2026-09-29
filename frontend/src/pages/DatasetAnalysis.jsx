import {
  useEffect,
  useMemo,
  useState
} from "react";

import {
  useNavigate,
  useParams
} from "react-router-dom";

import {
  getDataset
} from "../services/datasetApi";

import TargetValidation from "./TargetValidation";


function DatasetAnalysis() {

  const {
    datasetId
  } = useParams();

  const navigate = useNavigate();


  const [dataset, setDataset] =
    useState(null);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");



  useEffect(() => {

    async function loadDataset() {

      try {

        setLoading(true);

        const result =
          await getDataset(datasetId);

        setDataset(result);

      } catch (err) {

        setError(
          err.response?.data?.detail ||
          "Unable to load dataset."
        );

      } finally {

        setLoading(false);
      }
    }


    loadDataset();

  }, [datasetId]);


  const qualityReport =
    dataset?.quality_report;


  const profile =
    qualityReport?.profile;


  const columnTypes =
    qualityReport?.column_types || {};


  const idCandidates =
    qualityReport?.id_candidates || [];


  const idColumnNames =
    useMemo(
      () => new Set(
        idCandidates.map(
          (item) => item.column
        )
      ),
      [idCandidates]
    );


  if (loading) {

    return (
      <div className="loading-page">

        <div className="loader" />

        <p>
          Loading dataset analysis...
        </p>

      </div>
    );
  }


  if (error && !dataset) {

    return (
      <div className="loading-page">

        <div className="error-box">
          {error}
        </div>

        <button
          className="primary-button"
          onClick={() => navigate("/")}
        >
          Back to Upload
        </button>

      </div>
    );
  }


  return (
    <div className="workspace-page">
      <main className="page-container">

        <section className="page-heading">

          <span className="eyebrow">
            STAGE 01 · DATASET ANALYSIS
          </span>

          <h2>
            {dataset.original_filename}
          </h2>

          <p>
            AutoML has completed the initial
            inspection of your dataset.
          </p>

        </section>


        {/* STATUS */}

        <section className="quality-status-card">

          <div>

            <span className="section-label">
              DATASET QUALITY
            </span>

            <h3>
              {qualityReport.status ===
                "clean"
                ? "Clean"
                : qualityReport.status ===
                  "usable_with_warnings"
                  ? "Usable with warnings"
                  : "Blocked"}
            </h3>

          </div>


          <div className="quality-counts">

            <div>
              <strong>
                {qualityReport.summary.errors}
              </strong>
              <span>Errors</span>
            </div>

            <div>
              <strong>
                {qualityReport.summary.warnings}
              </strong>
              <span>Warnings</span>
            </div>

            <div>
              <strong>
                {qualityReport.summary.information}
              </strong>
              <span>Information</span>
            </div>

          </div>

        </section>


        {/* OVERVIEW */}

        <section className="section">

          <div className="section-title">

            <div>

              <span className="section-label">
                DATASET OVERVIEW
              </span>

              <h3>
                Structure
              </h3>

            </div>

          </div>


          <div className="overview-grid">

            <div className="metric-card">

              <span>Rows</span>

              <strong>
                {profile.rows.toLocaleString()}
              </strong>

            </div>


            <div className="metric-card">

              <span>Columns</span>

              <strong>
                {profile.columns}
              </strong>

            </div>


            <div className="metric-card">

              <span>Memory</span>

              <strong>
                {profile.memory.mb} MB
              </strong>

            </div>


            <div className="metric-card">

              <span>Duplicate Rows</span>

              <strong>
                {profile.duplicate_rows.count}
              </strong>

            </div>

          </div>

        </section>


        {/* WARNINGS */}

        {qualityReport.warnings.length > 0 && (

          <section className="section">

            <div className="section-title">

              <div>

                <span className="section-label">
                  QUALITY REPORT
                </span>

                <h3>
                  Warnings
                </h3>

              </div>

            </div>


            <div className="issue-list">

              {qualityReport.warnings.map(
                (warning, index) => (

                  <div
                    className="issue-card warning"
                    key={`${warning.code}-${index}`}
                  >
<div>

                      <strong>
                        {warning.code.replace(
                          /_/g,
                          " "
                        )}
                      </strong>

                      <p>
                        {warning.message}
                      </p>

                      {warning.suggested_action && (

                        <small>
                          {warning.suggested_action}
                        </small>
                      )}

                    </div>

                  </div>
                )
              )}

            </div>

          </section>
        )}


        {/* COLUMN ANALYSIS */}

        <section className="section">

          <div className="section-title">

            <div>

              <span className="section-label">
                COLUMN ANALYSIS
              </span>

              <h3>
                Detected Features
              </h3>

            </div>

            <span className="muted-text">
              {profile.columns} columns
            </span>

          </div>


          <div className="table-wrapper">

            <table>

              <thead>

                <tr>

                  <th>
                    Column
                  </th>

                  <th>
                    Detected Type
                  </th>

                  <th>
                    Missing
                  </th>

                  <th>
                    Missing %
                  </th>

                  <th>
                    Unique
                  </th>

                  <th>
                    ID Candidate
                  </th>

                </tr>

              </thead>


              <tbody>

                {profile.column_names.map(
                  (column) => {

                    const missing =
                      profile.missing_values[
                      column
                      ];

                    const unique =
                      profile.unique_values[
                      column
                      ];

                    const isId =
                      idColumnNames.has(
                        column
                      );


                    return (

                      <tr key={column}>

                        <td>
                          <strong>
                            {column}
                          </strong>
                        </td>

                        <td>

                          <span className="type-badge">

                            {columnTypes[
                              column
                            ]}

                          </span>

                        </td>

                        <td>
                          {missing.count}
                        </td>

                        <td>
                          {missing.percentage}%
                        </td>

                        <td>
                          {unique}
                        </td>

                        <td>

                          {isId ? (

                            <span className="id-badge">
                              Likely ID
                            </span>

                          ) : (

                            <span className="normal-badge">
                              —
                            </span>
                          )}

                        </td>

                      </tr>

                    );
                  }
                )}

              </tbody>

            </table>

          </div>

        </section>


        {/* DESCRIPTIVE STATISTICS */}

        <section className="section">

          <div className="section-title">

            <div>

              <span className="section-label">
                DESCRIPTIVE STATISTICS
              </span>

              <h3>
                Column Statistics
              </h3>

            </div>

          </div>


          <div className="stats-grid">

            {Object.entries(
              profile.descriptive_statistics
            ).map(
              ([column, stats]) => (

                <div
                  className="stats-card"
                  key={column}
                >

                  <h4>
                    {column}
                  </h4>

                  <span className="type-badge">
                    {stats.type}
                  </span>


                  {stats.mean !== undefined && (

                    <div className="stats-row">

                      <span>
                        Mean
                      </span>

                      <strong>
                        {formatNumber(
                          stats.mean
                        )}
                      </strong>

                    </div>
                  )}


                  {stats.std !== undefined && (

                    <div className="stats-row">

                      <span>
                        Std
                      </span>

                      <strong>
                        {formatNumber(
                          stats.std
                        )}
                      </strong>

                    </div>
                  )}


                  {stats.min !== undefined && (

                    <div className="stats-row">

                      <span>
                        Min
                      </span>

                      <strong>
                        {formatNumber(
                          stats.min
                        )}
                      </strong>

                    </div>
                  )}


                  {stats.max !== undefined && (

                    <div className="stats-row">

                      <span>
                        Max
                      </span>

                      <strong>
                        {formatNumber(
                          stats.max
                        )}
                      </strong>

                    </div>
                  )}


                  {stats.top_values && (

                    <div className="top-values">

                      <span>
                        Top values
                      </span>

                      {stats.top_values.map(
                        (item) => (

                          <div
                            key={item.value}
                            className="top-value"
                          >

                            <span>
                              {item.value}
                            </span>

                            <strong>
                              {item.count}
                            </strong>

                          </div>
                        )
                      )}

                    </div>
                  )}

                </div>
              )
            )}

          </div>

        </section>


        {/* PREVIEW */}

        <section className="section">

          <div className="section-title">

            <div>

              <span className="section-label">
                DATA PREVIEW
              </span>

              <h3>
                First 10 Rows
              </h3>

            </div>

          </div>


          <div className="table-wrapper">

            <table>

              <thead>

                <tr>

                  {profile.column_names.map(
                    (column) => (

                      <th key={column}>
                        {column}
                      </th>

                    )
                  )}

                </tr>

              </thead>


              <tbody>

                {profile.preview.map(
                  (row, index) => (

                    <tr key={index}>

                      {profile.column_names.map(
                        (column) => (

                          <td
                            key={`${index}-${column}`}
                          >
                            {formatCell(
                              row[column]
                            )}
                          </td>

                        )
                      )}

                    </tr>
                  )
                )}

              </tbody>

            </table>

          </div>

        </section>


        {/* TARGET VALIDATION — SAME WORKSPACE */}

        <TargetValidation
          embedded
          dataset={dataset}
        />

      </main>
    </div>
  );
}


function formatNumber(value) {

  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value !== "number") {
    return value;
  }

  return Number.isInteger(value)
    ? value
    : value.toFixed(3);
}


function formatCell(value) {

  if (
    value === null ||
    value === undefined
  ) {
    return "—";
  }

  return String(value);
}


export default DatasetAnalysis;