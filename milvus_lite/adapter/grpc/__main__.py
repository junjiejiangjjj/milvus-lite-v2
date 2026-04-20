"""``python -m milvus_lite.adapter.grpc`` entry point."""

import sys

from milvus_lite.adapter.grpc.cli import main

if __name__ == "__main__":
    sys.exit(main())
