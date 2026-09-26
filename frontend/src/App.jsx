import {
    BrowserRouter,
    Navigate,
    Route,
    Routes
} from "react-router-dom";

import UploadDataset from "./pages/UploadDataset";
import DatasetAnalysis from "./pages/DatasetAnalysis";
import TargetValidation from "./pages/TargetValidation";

function App() {

    return (
        <BrowserRouter>

            <Routes>

                {/* Module 1 - Upload */}
                <Route
                    path="/"
                    element={<UploadDataset />}
                />

                {/* Module 1 - Dataset Analysis */}
                <Route
                    path="/datasets/:datasetId"
                    element={<DatasetAnalysis />}
                />

                {/* Module 2 - Target Validation */}
                <Route
                    path="/datasets/:datasetId/target"
                    element={<TargetValidation />}
                />

                {/* Old route - keep temporarily */}
                <Route
                    path="/target"
                    element={<TargetValidation />}
                />

                {/* Unknown routes */}
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