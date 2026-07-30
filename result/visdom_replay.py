from __future__ import annotations

import argparse
from pathlib import Path


def replay(
    run_directory: str | Path,
    *,
    server: str = "http://localhost",
    port: int = 8097,
    base_url: str = "/",
) -> None:
    try:
        from visdom import Visdom
    except ImportError as error:
        raise RuntimeError(
            "Visdom is required for replay; install visdom==0.2.4"
        ) from error
    event_log = Path(run_directory) / "visdom_events.log"
    if not event_log.exists():
        raise FileNotFoundError(f"Visdom event log not found: {event_log}")
    client = Visdom(
        server=server,
        port=int(port),
        base_url=base_url,
        raise_exceptions=True,
        use_incoming_socket=False,
    )
    if not client.check_connection(timeout_seconds=2.0):
        raise ConnectionError(
            f"Visdom server is unavailable at {server}:{port}"
        )
    client.replay_log(str(event_log))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a training run's Visdom event log"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--server", default="http://localhost")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--base-url", default="/")
    args = parser.parse_args()
    replay(
        args.run_dir,
        server=args.server,
        port=args.port,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
