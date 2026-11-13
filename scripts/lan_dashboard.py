"""Make the dashboard reachable from phones on the home Wi-Fi (user request 2026-10-25). Home network only.

The API keeps listening on 127.0.0.1:8000 (scripts/supervise.py "api"). This process listens on port 8000 of the
other interfaces (0.0.0.0; Windows hands loopback connections to the more specific 127.0.0.1 socket, so the API
keeps those) and forwards each TCP connection to 127.0.0.1:8000 unchanged (pages, JSON, the /ws WebSocket). The API
is read-only and serves only numbers, density-only heatmaps and outlines, so nothing else leaves the PC.

Who can connect: Windows Firewall decides. scripts/allow_dashboard_lan.ps1 (run once as admin) sets the Wi-Fi to the
Private profile and allows inbound TCP 8000 on the Private profile only, from the local subnet only; on Public
networks it stays blocked. As a second guard this process drops peers outside private address ranges. No port
forwarding, no tunnels: nothing here reaches the internet.

Started by the supervisor (child "lan") from its next start; until then started by hand. Single instance.
  .venv/Scripts/python scripts/lan_dashboard.py      phone URL: http://<this PC's Wi-Fi IP>:8000
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.runtime import ALREADY_RUNNING_EXIT, setup_logging, single_instance  # noqa: E402

log = logging.getLogger("lan")
LISTEN = ("0.0.0.0", 8000)
TARGET = ("127.0.0.1", 8000)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()


async def handle(c_reader: asyncio.StreamReader, c_writer: asyncio.StreamWriter) -> None:
    peer = c_writer.get_extra_info("peername")[0]
    if not ipaddress.ip_address(peer).is_private:  # the firewall should already have dropped it
        log.warning("dropped a connection from %s (not a private address)", peer)
        c_writer.close()
        return
    try:
        a_reader, a_writer = await asyncio.open_connection(*TARGET)
    except OSError:  # the API is restarting: the browser retries
        c_writer.close()
        return
    await asyncio.gather(pipe(c_reader, a_writer), pipe(a_reader, c_writer))


async def serve() -> None:
    server = await asyncio.start_server(handle, *LISTEN)
    log.info("dashboard on the LAN: %s:%d -> %s:%d", *LISTEN, *TARGET)
    async with server:
        await server.serve_forever()


def main() -> None:
    if not single_instance("lan"):
        sys.exit(ALREADY_RUNNING_EXIT)
    setup_logging("lan")
    asyncio.run(serve())


if __name__ == "__main__":
    main()
