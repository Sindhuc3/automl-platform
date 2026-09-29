
const API_BASE_URL = "http://127.0.0.1:8000/api";

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Preprocessing request failed."
    );
  }

  return data;
}

export async function runPreprocessing(datasetId, options = {}) {
  const data = await request(
    `${API_BASE_URL}/datasets/${datasetId}/preprocessing`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: options.mode || "automatic",
        version: options.version || 1,
        overrides: options.overrides || {},
      }),
    }
  );

  return data.preprocessing;
}

export async function getLatestPreprocessing(datasetId) {
  const data = await request(
    `${API_BASE_URL}/datasets/${datasetId}/preprocessing`
  );
  return data.preprocessing || [];
}

export function preprocessingDownloadUrl(datasetId, runId, artifact) {
  return `${API_BASE_URL}/datasets/${datasetId}/preprocessing/${runId}/download/${artifact}`;
}
