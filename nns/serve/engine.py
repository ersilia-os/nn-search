import subprocess
from pathlib import Path

path = Path(__file__).parent / "engine.sh"


def start():
  subprocess.run(["bash", path, "start"])


def stop():
  subprocess.run(["bash", path, "stop"])


def restart():
  subprocess.run(["bash", path, "restart"])


def upgrade():
  subprocess.run(["bash", path, "upgrade"])


def delete():
  subprocess.run(["bash", path, "delete"])
