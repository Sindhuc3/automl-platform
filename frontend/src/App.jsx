import {
  BrowserRouter,
  Navigate,
  Route,
  Routes
} from "react-router-dom";

import UploadDataset from "./pages/UploadDataset";
import DatasetAnalysis from "./pages/DatasetAnalysis";


function App() {

  return (
    <BrowserRouter>

      <Routes>

        <Route
          path="/"
          element={<UploadDataset />}
        />

        <Route
          path="/analysis/:datasetId"
          element={<DatasetAnalysis />}
        />

        <Route
          path="*"
          element={
            <Navigate
              to="/"
              replace
            />
          }
        />

      </Routes>

    </BrowserRouter>
  );
}


export default App;