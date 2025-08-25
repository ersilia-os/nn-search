import sys, time, uuid

from nns.fetch.utils import ApiClient, JobStatus
from nns.utils.helpers import fetch_schema_from_github
from nns.utils.vars import API_BASE
from nns.utils.logging import logger
from nns.fetch.dump import DumpLocalCache


class FetchCloudCache:
  def __init__(self, model_id: str, fetch_lcoal: bool = False, file_name: str = None):
    super().__init__()
    self.model_id = model_id
    self.fetch_lcoal = fetch_lcoal
    self.file_name = file_name
    self.request_id = None
    self.fetch_type = "all"
    self.schema = fetch_schema_from_github(self.model_id)
    assert self.schema is not None, "Model schema can not be fetched from github."
    self.cols = self.schema[0]
    self.dtype = self.schema[1]
    self.header = ["key", "input"] + self.cols
    self.api = ApiClient()

  def build_vector_db(self) -> None:
    shards = self._submit_and_get_shards()
    self._merge_shards(shards, None)

  def _submit_and_get_shards(self) -> list:
    self.request_id = str(uuid.uuid4())

    logger.info(f"Submitting job for model_id={self.model_id} with all samples")

    payload = {
      "requestid": self.request_id,
      "modelid": self.model_id,
      "fetchtype": self.fetch_type,
      "nsamples": 10000,
      "dim": len(self.cols),
    }
    job_id = self.api.post_json(f"{API_BASE}/submit", json=payload)["jobId"]

    logger.info(f"Job submitted successfully. job_id={job_id}")

    start = time.time()
    while True:
      status = self.api.get_json(f"{API_BASE}/status", params={"jobId": job_id})["status"]

      logger.info(f"Job status: {status}")

      if status == JobStatus.SUCCEEDED:
        logger.info("Job succeeded.")
        break

      if status == JobStatus.FAILED:
        error = self.api.get_json(f"{API_BASE}/status", params={"jobId": job_id}).get(
          "errorMessage", "Unknown error"
        )
        logger.error(f"Job failed: {error}")
        raise RuntimeError(f"Job failed: {error}")

      if time.time() - start > 3600:
        logger.error("Job polling timed out.")
        raise RuntimeError("Job polling timed out")

      time.sleep(5)

    shards = self.api.get_json(f"{API_BASE}/result", params={"jobId": job_id})["files"]

    if not shards:
      logger.error(f"No shards returned for job_id={job_id}")
      raise RuntimeError("No shards returned")

    return shards

  def _merge_shards(self, shards: list) -> None:
    self.api.merge_shards(shards, self.api, self.model_id)

    logger.info(f"[green]Database has been successfully built for {self.model_id}![/]")

    sys.exit(1)
