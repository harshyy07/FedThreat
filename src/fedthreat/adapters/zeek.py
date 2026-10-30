"""Zeek / Bro conn.log adapter converting TSV or JSON log records to CanonicalEvents."""

import json
from datetime import datetime, timezone
from typing import List
from fedthreat.base import LogAdapter
from fedthreat.features.spec import CanonicalEvent


class ZeekAdapter(LogAdapter):
    """Adapter for Zeek network monitoring logs (e.g. conn.log)."""

    def parse_line(self, line: str) -> CanonicalEvent:
        """Parses a single Zeek conn.log JSON or TSV line."""
        line = line.strip()
        if not line or line.startswith("#"):
            return CanonicalEvent()

        # Handle JSON formatted Zeek logs
        if line.startswith("{"):
            data = json.loads(line)
            ts = data.get("ts")
            dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
            proto = str(data.get("proto", "tcp")).lower()
            duration_sec = float(data.get("duration", 0.0) or 0.0)
            orig_bytes = float(data.get("orig_bytes", 0.0) or 0.0)
            resp_bytes = float(data.get("resp_bytes", 0.0) or 0.0)
            orig_pkts = float(data.get("orig_pkts", 0.0) or 0.0)
            resp_pkts = float(data.get("resp_pkts", 0.0) or 0.0)
            conn_state = str(data.get("conn_state", ""))

            # Zeek does not log auth attempts in conn.log, so failed_auth_count is omitted/None
            return CanonicalEvent(
                timestamp=dt,
                src_ip=data.get("id.orig_h"),
                dst_ip=data.get("id.resp_h"),
                src_port=data.get("id.orig_p"),
                dst_port=data.get("id.resp_p"),
                protocol=proto,
                action="allow" if conn_state in ["SF", "S1", "S2"] else "drop",
                service=str(data.get("service", "unknown")),
                duration_ms=duration_sec * 1000.0,
                src_bytes=orig_bytes,
                dst_bytes=resp_bytes,
                total_packets=orig_pkts + resp_pkts,
                failed_auth_count=0.0,  # Or None
                extra_fields={"conn_state": conn_state},
            )

        # Handle standard Zeek tab-separated conn.log
        fields = line.split("\t")
        # Standard columns: ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p, proto, service, duration, orig_bytes, resp_bytes, conn_state, ...
        try:
            ts_val = float(fields[0]) if len(fields) > 0 else 0.0
            dt = datetime.fromtimestamp(ts_val, tz=timezone.utc) if ts_val > 0 else datetime.now(timezone.utc)
            src_ip = fields[2] if len(fields) > 2 else None
            src_p = int(fields[3]) if len(fields) > 3 and fields[3] != "-" else None
            dst_ip = fields[4] if len(fields) > 4 else None
            dst_p = int(fields[5]) if len(fields) > 5 and fields[5] != "-" else None
            proto = fields[6].lower() if len(fields) > 6 else "tcp"
            service = fields[7] if len(fields) > 7 and fields[7] != "-" else "unknown"
            duration = float(fields[8]) * 1000.0 if len(fields) > 8 and fields[8] != "-" else 0.0
            orig_b = float(fields[9]) if len(fields) > 9 and fields[9] != "-" else 0.0
            resp_b = float(fields[10]) if len(fields) > 10 and fields[10] != "-" else 0.0
            conn_state = fields[11] if len(fields) > 11 else ""

            return CanonicalEvent(
                timestamp=dt,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_p,
                dst_port=dst_p,
                protocol=proto,
                action="allow" if conn_state in ["SF", "S1", "S2"] else "drop",
                service=service,
                duration_ms=duration,
                src_bytes=orig_b,
                dst_bytes=resp_b,
                total_packets=0.0,
                failed_auth_count=0.0,
                extra_fields={"conn_state": conn_state},
            )
        except Exception:
            return CanonicalEvent()

    def parse_file(self, filepath: str) -> List[CanonicalEvent]:
        events = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                ev = self.parse_line(line)
                if ev.src_port is not None or ev.dst_port is not None or ev.duration_ms > 0:
                    events.append(ev)
        return events
