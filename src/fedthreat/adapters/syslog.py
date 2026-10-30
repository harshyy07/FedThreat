"""Syslog RFC 5424 / RFC 3164 adapter converting firewall/router logs into CanonicalEvents."""

import re
from datetime import datetime, timezone
from typing import List
from fedthreat.base import LogAdapter
from fedthreat.features.spec import CanonicalEvent


class SyslogAdapter(LogAdapter):
    """Adapter for standard Syslog firewall and connection events."""

    # Matches common key-value firewall syslog formats like Cisco ASA or iptables:
    # "Oct 26 12:00:00 fw kernel: [ALLOW] SRC=192.168.1.50 DST=10.0.0.1 PROTO=TCP SPT=49152 DPT=80 LEN=64"
    KV_REGEX = re.compile(r"([A-Za-z0-9_-]+)=([^\s]+)")

    def parse_line(self, line: str) -> CanonicalEvent:
        line = line.strip()
        if not line:
            return CanonicalEvent()

        kv_matches = dict(self.KV_REGEX.findall(line))
        
        # Action detection from message body
        action = "allow"
        line_lower = line.lower()
        if "drop" in line_lower or "deny" in line_lower or "block" in line_lower:
            action = "drop"

        # Protocol
        proto = kv_matches.get("PROTO", kv_matches.get("proto", "tcp")).lower()

        # Ports
        spt = kv_matches.get("SPT", kv_matches.get("spt", kv_matches.get("src_port")))
        dpt = kv_matches.get("DPT", kv_matches.get("dpt", kv_matches.get("dst_port")))
        src_port = int(spt) if spt and spt.isdigit() else None
        dst_port = int(dpt) if dpt and dpt.isdigit() else None

        # Bytes / Length
        length = float(kv_matches.get("LEN", kv_matches.get("bytes", 0)))

        return CanonicalEvent(
            timestamp=datetime.now(timezone.utc),
            src_ip=kv_matches.get("SRC", kv_matches.get("src")),
            dst_ip=kv_matches.get("DST", kv_matches.get("dst")),
            src_port=src_port,
            dst_port=dst_port,
            protocol=proto,
            action=action,
            service="firewall",
            duration_ms=0.0,
            src_bytes=length,
            dst_bytes=0.0,
            total_packets=1.0,
            failed_auth_count=0.0,
            extra_fields=kv_matches,
        )

    def parse_file(self, filepath: str) -> List[CanonicalEvent]:
        events = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                events.append(self.parse_line(line))
        return events
