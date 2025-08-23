REDIS_EXPIRATION = 3600 * 24 * 7
REDIS_PORT = 6379
REDIS_CONTAINER_NAME = "redis"
REDIS_IMAGE = "redis:latest"
REDIS_HOST = "127.0.0.1"
DEFAULT_API_NAME = "run"
S3_BUCKET_URL = "https://ersilia-models.s3.eu-central-1.amazonaws.com"
S3_BUCKET_URL_ZIP = "https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com"
INFERENCE_STORE_API_URL = (
  "https://5x2fkcjtei.execute-api.eu-central-1.amazonaws.com/dev/precalculations"
)
API_BASE = "https://hov95ejni7.execute-api.eu-central-1.amazonaws.com/dev/predict"
GITHUB_ORG = "ersilia-os"
GITHUB_CONTENT_URL = f"https://raw.githubusercontent.com/{GITHUB_ORG}"
GITHUB_ERSILIA_REPO = "ersilia"
PREDEFINED_COLUMN_FILE = "model/framework/columns/run_columns.csv"
