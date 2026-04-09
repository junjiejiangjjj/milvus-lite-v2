"""CLI entry point for the LiteVecDB gRPC server.

Usage:
    python -m litevecdb.adapter.grpc --data-dir ./data --port 19530

Or, after `pip install -e .[grpc]`:
    litevecdb-grpc --data-dir ./data --port 19530
    (the script entry is registered in pyproject.toml under
     [project.scripts])
"""

from __future__ import annotations

import argparse
import sys

from litevecdb.adapter.grpc.server import (
    DEFAULT_HOST,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PORT,
    run_server,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m litevecdb.adapter.grpc",
        description="Start the LiteVecDB gRPC server (Milvus protocol).",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="LiteVecDB data directory (created if it doesn't exist).",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind host (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Thread pool size (default: {DEFAULT_MAX_WORKERS}).",
    )
    args = parser.parse_args(argv)

    run_server(
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
