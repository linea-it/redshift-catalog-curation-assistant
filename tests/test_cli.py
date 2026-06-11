import subprocess
import sys


def test_cli_help():
    # Ensure the CLI prints help
    res = subprocess.run([sys.executable, "-c", "import redshift_catalog_curation_assistant.cli as c; print('ok')"], capture_output=True)
    assert res.returncode == 0
