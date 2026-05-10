"""
Lightweight HTTP status server — check bot status without logging into TopstepX.

Runs on http://localhost:8585
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

logger = logging.getLogger(__name__)

_bot_ref = None


def start_status_server(bot, port=8585):
    """Start status server in background thread."""
    global _bot_ref
    _bot_ref = bot
    
    server = HTTPServer(("127.0.0.1", port), StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Status server running at http://localhost:{port}")


class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        bot = _bot_ref
        if not bot:
            self._respond(500, {"error": "Bot not initialized"})
            return
        
        if self.path == "/api/status":
            self._respond(200, self._build_status(bot))
        elif self.path == "/":
            self._respond_html(self._build_dashboard(bot))
        else:
            self._respond(404, {"error": "Not found"})
    
    def _build_status(self, bot):
        summary = bot.risk.get_summary()
        platform_pnl = getattr(bot, 'platform_pnl', {})
        sod = getattr(bot, 'sod_balance', None)
        bal = bot.account_balance or 0
        # Primary: balance-based daily P&L (most accurate — survives restarts, synced every 5s)
        # Fallback: platform realizedPnl if SOD balance not available yet
        balance_based_pnl = (bal - sod) if (sod and bal) else None
        daily_pnl = balance_based_pnl if balance_based_pnl is not None else summary["daily_pnl"]
        return {
            "connected": bot.conn.is_connected,
            "account_balance": bot.account_balance,
            "daily_pnl": daily_pnl,
            "local_pnl": summary["daily_pnl"],
            "platform_realized_pnl": platform_pnl.get("realizedPnl", 0),
            "platform_unrealized_pnl": platform_pnl.get("unrealizedPnl", 0),
            "platform_total_pnl": platform_pnl.get("totalPnl", 0),
            "platform_open_pnl": platform_pnl.get("openPnl", 0),
            "total_trades": summary["total_trades"],
            "winners": summary["winners"],
            "losers": summary["losers"],
            "win_rate": f"{summary['win_rate']:.1%}",
            "consecutive_losses": summary["consecutive_losses"],
            "bars_processed": bot._bar_count,
            "quotes_received": bot._quote_count,
            "signals_generated": bot._signal_count,
            "position": {
                "direction": bot.position_direction.name if bot.position_direction else "FLAT",
                "entry_price": bot.position_entry_price,
                "contracts": bot.position_contracts,
                "strategy": bot.position_strategy,
            } if bot.position_direction else None,
            "price": bot.features.last_price,
            "vwap": round(bot.features.vwap(), 2) if bot.features.bar_count > 0 else 0,
            "atr": round(bot.features.atr(), 2) if bot.features.bar_count > 0 else 0,
            "rsi": round(bot.features.rsi(), 1) if bot.features.bar_count > 0 else 0,
            "instrument": getattr(bot.instrument, "instrument", "BTC"),
        }
    
    def _build_dashboard(self, bot):
        s = self._build_status(bot)
        pos = s["position"]
        pos_html = "<span style='color:#4CAF50'>FLAT</span>" if not pos else (
            f"<span style='color:{'#4CAF50' if pos['direction']=='LONG' else '#f44336'}'>"
            f"{pos['direction']} {pos['contracts']}x @ {pos['entry_price']:.2f} ({pos['strategy']})</span>"
        )
        
        pnl_color = "#4CAF50" if s.get("daily_pnl", 0) >= 0 else "#f44336"
        conn_color = "#4CAF50" if s["connected"] else "#f44336"
        
        return f"""<!DOCTYPE html><html><head><title>Topstep5 Bot</title>
<meta http-equiv="refresh" content="5">
<style>
body {{ font-family: 'Consolas', monospace; background: #1a1a2e; color: #eee; padding: 30px; }}
.card {{ background: #16213e; border-radius: 8px; padding: 20px; margin: 10px 0; }}
h1 {{ color: #0f3460; }} h2 {{ color: #e94560; margin: 0 0 10px 0; }}
.metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
.metric .value {{ font-size: 28px; font-weight: bold; }}
.metric .label {{ font-size: 12px; color: #888; }}
</style></head><body>
<h1>⚡ TOPSTEP5 — Live Dashboard</h1>
<div class="card">
<h2>Connection</h2>
<div class="metric"><div class="value" style="color:{conn_color}">{'● CONNECTED' if s['connected'] else '○ DISCONNECTED'}</div><div class="label">SignalR</div></div>
<div class="metric"><div class="value">{s['quotes_received']}</div><div class="label">Quotes</div></div>
<div class="metric"><div class="value">{s['bars_processed']}</div><div class="label">Bars</div></div>
</div>
<div class="card">
<h2>Market</h2>
<div class="metric"><div class="value">{s['price']:.2f}</div><div class="label">{s.get('instrument', 'BTC')} Price</div></div>
<div class="metric"><div class="value">{s['vwap']}</div><div class="label">VWAP</div></div>
<div class="metric"><div class="value">{s['atr']}</div><div class="label">ATR(14)</div></div>
<div class="metric"><div class="value">{s['rsi']}</div><div class="label">RSI(14)</div></div>
</div>
<div class="card">
<h2>Trading</h2>
<div class="metric"><div class="value" style="color:{pnl_color}">${s['daily_pnl']:+,.2f}</div><div class="label">Daily P&L (Balance-Based)</div></div>
<div class="metric"><div class="value">${s['platform_unrealized_pnl']:+,.2f}</div><div class="label">Unrealized</div></div>
<div class="metric"><div class="value">${s['local_pnl']:+,.2f}</div><div class="label">Bot-Tracked P&L</div></div>
<div class="metric"><div class="value">{s['total_trades']}</div><div class="label">Trades</div></div>
<div class="metric"><div class="value">{s['win_rate']}</div><div class="label">Win Rate</div></div>
<div class="metric"><div class="value">{s['signals_generated']}</div><div class="label">Signals</div></div>
</div>
<div class="card">
<h2>Position</h2>
<div class="metric"><div class="value">{pos_html}</div><div class="label">Current</div></div>
</div>
<div class="card">
<h2>Account</h2>
<div class="metric"><div class="value">${s['account_balance']:,.2f}</div><div class="label">Balance</div></div>
</div>
<p style="color:#555;font-size:11px">Auto-refreshes every 5 seconds</p>
</body></html>"""
    
    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())
    
    def _respond_html(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP request logging
