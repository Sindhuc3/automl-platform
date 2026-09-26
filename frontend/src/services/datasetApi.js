import axios from "axios";


const api = axios.create({
  baseURL: "/api",
  headers: {
    Accept: "application/json"
  }
});


export async function uploadDataset(file, onUploadProgress) {

  const formData = new FormData();

  formData.append("file", file);

  const response = await api.post(
    "/datasets/upload",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data"
      },

      onUploadProgress
    }
  );

  return response.data;
}


export async function getDataset(datasetId) {

  const response = await api.get(
    `/datasets/${datasetId}`
  );

  return response.data;
}


export async function getQualityReport(datasetId) {

  const response = await api.get(
    `/datasets/${datasetId}/quality-report`
  );

  return response.data;
}


export async function getDatasets() {

  const response = await api.get(
    "/datasets"
  );

  return response.data;
}


export async function saveTargetColumn(
  datasetId,
  targetColumn
) {

  const response = await api.post(
    `/datasets/${datasetId}/target`,
    {
      target_column: targetColumn
    }
  );

  return response.data;
}