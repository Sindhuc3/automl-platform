function Runs() {
  return (
    <div className="utility-page">
      <p className="page-eyebrow">RUNS</p>
      <h1>Pipeline runs</h1>
      <p className="utility-lead">
        Previous automatic and customized runs will appear here after
        authentication and persistent run storage are connected.
      </p>
      <div className="utility-panel">
        <h2>Guest workspace</h2>
        <p>
          Current guest sessions can run AutoML and download generated
          outputs, but runs are not added to a personal history.
        </p>
      </div>
    </div>
  );
}
export default Runs;
