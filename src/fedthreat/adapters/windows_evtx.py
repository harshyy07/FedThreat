"""Windows Security Event Log adapter (Event ID 4624/4625 for Logon success/failure, 5156 for Filtering Platform)."""

import json
from datetime import datetime, timezone
from typing import List
from fedthreat.base import LogAdapter
from fedthreat.features.spec import CanonicalEvent


class WindowsEvtxAdapter(LogAdapter):
    """Adapter for Windows Security Events (parsed from JSON or XML representations)."""

    def parse_line(self, line: str) -> CanonicalEvent:
        """Parses a single Windows Security Event (JSON representation)."""
        line = line.strip()
        if not line:
            return CanonicalEvent()

        try:
            data = json.loads(line)
        except Exception:
            return CanonicalEvent()

        event_id = int(data.get("EventID", 0))
        event_data = data.get("EventData", {})

        # Default timestamp
        time_created = data.get("TimeCreated", "")
        try:
            dt = datetime.fromisoformat(time_created.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.now(timezone.utc)

        # Event ID 4625: An account failed to log on (Brute force indicator)
        if event_id == 4625:
            ip = event_data.get("IpAddress", "0.0.0.0")
            port = event_data.get("IpPort", None)
            port_int = int(port) if port and port != "-" else 3389
            return CanonicalEvent(
                timestamp=dt,
                src_ip=ip,
                dst_port=port_int,
                protocol="tcp",
                action="drop",
                service="auth",
                duration_ms=50.0,
                src_bytes=200.0,
                dst_bytes=100.0,
                total_packets=4.0,
                failed_auth_count=1.0,
                extra_fields={"event_id": event_id, "target_user": event_data.get("TargetUserName")},
            )

        # Event ID 4624: An account was successfully logged on
        elif event_id == 4624:
            ip = event_data.get("IpAddress", "127.0.0.1")
            port = event_data.get("IpPort", None)
            port_int = int(port) if port and port != "-" else 3389
            return CanonicalEvent(
                timestamp=dt,
                src_ip=ip,
                dst_port=port_int,
                protocol="tcp",
                action="allow",
                service="auth",
                duration_ms=100.0,
                src_bytes=500.0,
                dst_bytes=800.0,
                total_packets=8.0,
                failed_auth_count=0.0,
                extra_fields={"event_id": event_id, "target_user": event_data.get("TargetUserName")},
            )

        # Event ID 5156: Windows Filtering Platform has permitted a connection
        elif event_id == 5156:
            src_ip = event_data.get("SourceAddress")
            src_port = int(event_data.get("SourcePort", 0)) if event_data.get("SourcePort") else None
            dst_ip = event_data.get("DestAddress")
            dst_port = int(event_data.get("DestPort", 0)) if event_data.get("DestPort") else None
            proto_num = int(event_data.get("Protocol", 6))
            proto_str = "tcp" if proto_num == 6 else ("udp" if proto_num == 17 else "icmp")

            return CanonicalEvent(
                timestamp=dt,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=proto_str,
                action="allow",
                service="unknown",
                duration_ms=0.0,  # Windows firewall events do not track session duration
                src_bytes=0.0,
                dst_bytes=0.0,
                total_packets=1.0,
                failed_auth_count=0.0,
                extra_fields={"app": event_data.get("Application")},
            )

        return CanonicalEvent(timestamp=dt, extra_fields={"event_id": event_id})

    def parse_file(self, filepath: str) -> List[CanonicalEvent]:
        events = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                ev = self.parse_line(line)
                events.append(ev)
        return events
