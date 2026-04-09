"""``python -m litevecdb.adapter.grpc`` entry point."""

import sys

from litevecdb.adapter.grpc.cli import main

if __name__ == "__main__":
    sys.exit(main())
