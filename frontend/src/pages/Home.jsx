import { useNavigate } from "react-router-dom";

function Home() {
  const navigate = useNavigate();

  return (
    <div className="home-page">
      <section className="home-hero">
        <div className="hero-copy">
          <p className="hero-eyebrow">AUTOMATED MACHINE LEARNING</p>
          <h1>
            From dataset
            <br />
            to a working model.
          </h1>
          <p className="hero-description">
            Upload your dataset and let AutoML Studio inspect the data,
            build a machine learning pipeline, evaluate the results,
            and keep every decision understandable.
          </p>
          <button
            className="hero-button"
            onClick={() => navigate("/upload")}
          >
            Upload Dataset
          </button>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <img src="/hero-analytics.png" alt="" />
        </div>
      </section>

      <section className="home-section">
        <div className="section-intro">
          <p className="home-eyebrow">HOW IT WORKS</p>
          <h2>One workspace. Four stages.</h2>
          <p>
            The workflow moves from understanding the dataset to a tested
            model without turning each stage into a separate page.
          </p>
        </div>

        <div className="workflow-grid">
          <article className="workflow-item">
            <span>01</span>
            <h3>Dataset Analysis</h3>
            <p>
              Structure, quality, detected types, missing values,
              identifiers and column-wise analysis.
            </p>
          </article>

          <article className="workflow-item">
            <span>02</span>
            <h3>Target Validation</h3>
            <p>
              You select the target. AutoML checks whether it is usable
              and determines how it should be modeled.
            </p>
          </article>

          <article className="workflow-item">
            <span>03</span>
            <h3>Automatic Pipeline</h3>
            <p>
              Preprocessing through evaluation runs together as one
              automatic machine learning pipeline.
            </p>
          </article>

          <article className="workflow-item">
            <span>04</span>
            <h3>Guided Customization</h3>
            <p>
              Only after the first run can you edit supported stages
              and create another pipeline version.
            </p>
          </article>
        </div>
      </section>

      <section className="guest-panel">
        <div>
          <p className="home-eyebrow">GUEST WORKSPACE</p>
          <h2>Try the workflow first.</h2>
        </div>
        <div className="guest-copy">
          <p>
            Guest sessions can upload, analyze, run AutoML, and download
            outputs. Persistent datasets and previous runs will be
            available after authentication is connected.
          </p>
          <p>
            Nothing is added to a personal dataset library while you are
            a guest.
          </p>
        </div>
      </section>
    </div>
  );
}

export default Home;
