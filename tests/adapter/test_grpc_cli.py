"""CLI entry point tests for the gRPC server."""

from __future__ import annotations

import runpy
import sys

import pytest

from milvus_lite.adapter.grpc import cli


def test_cli_main_forwards_parsed_arguments(monkeypatch, tmp_path):
    calls = []

    def fake_run_server(data_dir, host, port, max_workers):
        calls.append({
            "data_dir": data_dir,
            "host": host,
            "port": port,
            "max_workers": max_workers,
        })

    monkeypatch.setattr(cli, "run_server", fake_run_server)

    rc = cli.main([
        "--data-dir", str(tmp_path / "data"),
        "--host", "127.0.0.1",
        "--port", "19531",
        "--max-workers", "3",
    ])

    assert rc == 0
    assert calls == [{
        "data_dir": str(tmp_path / "data"),
        "host": "127.0.0.1",
        "port": 19531,
        "max_workers": 3,
    }]


def test_cli_requires_data_dir():
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code == 2


def test_grpc_module_main_delegates_to_cli_main(monkeypatch):
    calls = []

    def fake_main():
        calls.append(True)
        return 17

    monkeypatch.setattr(cli, "main", fake_main)

    # Ensure __main__ imports the already-patched cli module.
    monkeypatch.setitem(sys.modules, "milvus_lite.adapter.grpc.cli", cli)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("milvus_lite.adapter.grpc.__main__", run_name="__main__")

    assert exc_info.value.code == 17
    assert calls == [True]
