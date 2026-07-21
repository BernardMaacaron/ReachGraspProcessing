import subprocess
import sys


def test_package_initializer_does_not_eagerly_import_plotting():
    command = [
        sys.executable,
        "-c",
        "import sys, reachgrasp.data_loading; "
        "assert 'reachgrasp.plotting' not in sys.modules; "
        "assert 'matplotlib.pyplot' not in sys.modules",
    ]
    subprocess.run(command, check=True)
