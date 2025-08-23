import rich_click as click
import rich_click.rich_click as rc
import os
from nns.utils.helpers import summerize_status
from nns.fetch.fetch import FetchCloudCache
from nns.serve.engine import start as start_engine
from nns.serve.engine import restart as restart_engine
from nns.serve.engine import stop as stop_engine
from nns.serve.engine import delete as delete_engine
from nns.serve.engine import upgrade as upgrade_engine
from nns.filter.retrieve import ann_from_csv_to_json

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True


rc.USE_RICH_MARKUP = True
rc.SHOW_ARGUMENTS = True
rc.COLOR_SYSTEM = "truecolor"
rc.STYLE_OPTION = "bold magenta"
rc.STYLE_COMMAND = "bold green"
rc.STYLE_METAVAR = "italic yellow"
rc.STYLE_SWITCH = "underline cyan"
rc.STYLE_USAGE = "bold blue"
rc.STYLE_OPTION_DEFAULT = "dim italic"


def common_options(func):
  options = [
    click.option("--input-file", "-i", required=True, help="Path to the input file."),
    click.option(
      "--model-dir",
      "-m",
      required=False,
      default=None,
      help="Directory where the model is stored.",
    ),
    click.option("--anonymize", is_flag=True, help="Whether to anonymize outputs."),
  ]
  for option in reversed(options):
    func = option(func)
  return func


@click.group()
def cli():
  pass


@cli.command()
@click.option("--input-file", "-i", required=True, help="Path to the input file.")
@click.option("--output-file", "-o", required=True, help="Path to the output file.")
@click.option(
  "--collection-name", "-c", required=True, help="Collection name of the vector db"
)
@click.option("--topk", "-k", required=False, default=5, type=int, help="Top k results")
def filter(input_file, output_file, collection_name, topk):
  ann_from_csv_to_json(
    input_csv=input_file, output_path=output_file, collection=collection_name, topk=topk
  )


@cli.command()
@click.option("--start", "-s", is_flag=True, default=False)
@click.option("--restart", "-r", is_flag=True, default=False)
@click.option("--stop", "-p", is_flag=True, default=False)
@click.option("--upgrade", "-u", is_flag=True, default=False)
@click.option("--delete", "-d", is_flag=True, default=False)
def engine(start, restart, stop, upgrade, delete):
  if start:
    start_engine()
  if restart:
    restart()
  if stop:
    stop_engine()
  if upgrade:
    upgrade_engine()
  if delete:
    delete()


@cli.command()
@click.option(
  "--milvus-uri",
  default=os.getenv("MILVUS_URI", "http://localhost:19530"),
  show_default=True,
  help="Milvus URI (e.g., http://localhost:19530)",
)
@click.option(
  "--alias", default="default", show_default=True, help="Milvus connection alias"
)
@click.option("--token", default=None, help="Auth token, e.g. 'root:Milvus'")
@click.option(
  "--timeout",
  default=1.5,
  show_default=True,
  type=float,
  help="TCP connect timeout (seconds)",
)
@click.option(
  "--show-indexes/--no-show-indexes",
  default=True,
  show_default=True,
  help="Show index details",
)
def status(milvus_uri, alias, token, timeout, show_indexes):
  summerize_status(milvus_uri, alias, token, timeout, show_indexes)


@cli.command()
@click.option(
  "--collection-name", "-c", required=True, help="Collection name of the vector db"
)
def build(collection_name):
  fcc = FetchCloudCache(collection_name)
  fcc.build_vector_db()


if __name__ == "__main__":
  cli()
