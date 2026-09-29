import {
  BrowserRouter,
  Navigate,
  Outlet,
  Route,
  Routes,
  useParams
} from "react-router-dom";

import AppShell from "./components/AppShell";
import Home from "./pages/Home";
import UploadDataset from "./pages/UploadDataset";
import DatasetAnalysis from "./pages/DatasetAnalysis";
import TargetValidation from "./pages/TargetValidation";
import Datasets from "./pages/Datasets";
import Runs from "./pages/Runs";
import Settings from "./pages/Settings";
import PreprocessingWorkspace from "./pages/PreprocessingWorkspace";

function ShellLayout() {
  return <AppShell />;
}

function NavigateToDataset() {
  const { datasetId } = useParams();
  return <Navigate to={`/datasets/${datasetId}`} replace />;
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<ShellLayout />}>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<UploadDataset />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/datasets/:datasetId" element={<DatasetAnalysis />} />
          <Route path="/datasets/:datasetId/target" element={<NavigateToDataset />} />
          <Route path="/datasets/:datasetId/preprocessing" element={<PreprocessingWorkspace />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/settings" element={<Settings />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
