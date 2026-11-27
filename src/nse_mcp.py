"""Lightweight HTTP JSON-RPC client for official NSE India MCP servers."""

from __future__ import annotations

import json
from typing import Any

import requests

BHAVCOPY_MCP_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"
CMMKT_MCP_URL = "https://mcp.nseindia.in/cmmkt/mcp"


class NseMcpClient:
    """Client for NSE Bhavcopy and Capital Market Streamable HTTP MCP endpoints."""

    def __init__(self, endpoint: str = BHAVCOPY_MCP_URL, timeout: float = 15.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._session = requests.Session()
        self._session_id: str | None = None
        self._req_id = 0
        self._cmmkt_client: NseMcpClient | None = None

    def _cmmkt(self) -> NseMcpClient:
        if self.endpoint == CMMKT_MCP_URL:
            return self
        if self._cmmkt_client is None:
            self._cmmkt_client = NseMcpClient(endpoint=CMMKT_MCP_URL, timeout=self.timeout)
        return self._cmmkt_client

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            h["mcp-session-id"] = self._session_id
        return h

    @staticmethod
    def _parse_rpc_body(text: str) -> dict[str, Any]:
        for line in text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        return json.loads(text)

    def connect(self) -> dict[str, Any]:
        self._req_id += 1
        resp = self._session.post(
            self.endpoint,
            json={
                "jsonrpc": "2.0",
                "id": self._req_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "midcap-momentum-index", "version": "1.0"},
                },
            },
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        self._session_id = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
        self._session.post(
            self.endpoint,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=self._headers(),
            timeout=self.timeout,
        )
        return self._parse_rpc_body(resp.text)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._session_id:
            self.connect()
        self._req_id += 1
        resp = self._session.post(
            self.endpoint,
            json={
                "jsonrpc": "2.0",
                "id": self._req_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        parsed = self._parse_rpc_body(resp.text)
        result = parsed.get("result") or {}
        if result.get("isError"):
            texts = [c.get("text", "") for c in result.get("content", [])]
            raise RuntimeError(f"NSE MCP {name} failed: {' '.join(texts)}")
        content = result.get("content") or []
        if content and content[0].get("type") == "text":
            txt = content[0]["text"]
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                return txt
        return result

    def get_market_mood(self, date: str = "today") -> dict[str, Any]:
        return self.call_tool("get_market_mood", {"date": date})

    def get_market_breadth(self, date: str = "today") -> dict[str, Any]:
        return self.call_tool("get_market_breadth", {"date": date})

    def get_index_valuation(self, index_name: str = "Nifty Midcap 150", months: int = 24, date: str = "today") -> dict[str, Any]:
        return self.call_tool("get_index_valuation", {"indexName": index_name, "months": months, "date": date})

    def get_index_history(self, index_name: str = "Nifty Midcap 150", months: int = 12, end_date: str = "today") -> dict[str, Any]:
        return self.call_tool("get_index_history", {"indexName": index_name, "months": months, "endDate": end_date})

    def get_bulk_quote(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch quotes for up to 50 NSE symbols per call."""
        all_rows: dict[str, Any] = {}
        for i in range(0, len(symbols), 50):
            chunk = symbols[i : i + 50]
            res = self.call_tool("get_bulk_quote", {"symbols": chunk})
            if isinstance(res, dict):
                rows = res.get("quotes") or res.get("data") or res.get("results")
                if isinstance(rows, list):
                    for item in rows:
                        if isinstance(item, dict) and item.get("symbol"):
                            all_rows[item["symbol"]] = item
                else:
                    all_rows[f"chunk_{i}"] = res
        return all_rows

    def get_corporate_actions(self, symbol: str) -> dict[str, Any]:
        return self.call_tool("get_corporate_actions", {"symbol": symbol})

    def get_stock_history(self, symbol: str, months: int = 3, end_date: str = "today") -> dict[str, Any]:
        return self.call_tool("get_stock_history", {"symbol": symbol, "months": months, "endDate": end_date})

    def get_ltp_by_date(self, symbol: str, date: str = "today") -> dict[str, Any]:
        return self.call_tool("get_ltp_by_date", {"symbol": symbol, "date": date})

    def cm_get_stock_quote(self, symbol: str) -> dict[str, Any]:
        return self._cmmkt().call_tool("cm_get_stock_quote", {"symbol": symbol})

    def cm_get_index_quote(self, index_name: str = "NIFTY MIDCAP 150") -> dict[str, Any]:
        return self._cmmkt().call_tool("cm_get_index_quote", {"indexName": index_name})
