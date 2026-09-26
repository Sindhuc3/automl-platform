const API_BASE_URL =
  "http://127.0.0.1:8000/api";


export async function validateTarget(
  datasetId,
  targetColumn
) {

  const response = await fetch(
    `${API_BASE_URL}/datasets/${datasetId}/target`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        target_column: targetColumn,
      }),
    }
  );


  const data =
    await response.json();


  console.log(
    "Target validation response:",
    data
  );


  if (!response.ok) {

    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Target validation failed."
    );

  }


  /*
   * Backend normally returns:
   *
   * {
   *   message: "...",
   *   dataset_id: "...",
   *   target: {...}
   * }
   *
   * The fallback also makes the frontend
   * tolerant if the backend returns the
   * target object directly.
   */

  const targetResult =
    data.target || data;


  if (!targetResult.status) {

    throw new Error(
      "The backend returned an invalid target validation response."
    );

  }


  return targetResult;
}


export async function getTarget(
  datasetId
) {

  const response = await fetch(
    `${API_BASE_URL}/datasets/${datasetId}/target`
  );


  const data =
    await response.json();


  if (!response.ok) {

    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Unable to retrieve target."
    );

  }


  return data;
}