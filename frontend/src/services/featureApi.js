const API_BASE_URL = "http://127.0.0.1:8000/api";
async function request(url, options={}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Feature engineering request failed.");
  return data;
}
export async function runFeatureEngineering(datasetId, options={}) {
  const data=await request(`${API_BASE_URL}/datasets/${datasetId}/feature-engineering`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode:options.mode||"automatic",version:options.version||1,preprocessing_run_id:options.preprocessingRunId||null,overrides:options.overrides||{}})});
  return data.feature_engineering;
}
export async function getLatestFeatureEngineering(datasetId){
  const data=await request(`${API_BASE_URL}/datasets/${datasetId}/feature-engineering`); return data.feature_engineering||[];
}
export function featureDownloadUrl(datasetId,runId,artifact){return `${API_BASE_URL}/datasets/${datasetId}/feature-engineering/${runId}/download/${artifact}`;}
