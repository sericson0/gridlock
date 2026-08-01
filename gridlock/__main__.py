"""Allow ``python -m gridlock ...`` as an alternative to the ``gridlock`` script."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
