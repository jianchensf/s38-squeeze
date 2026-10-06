"""Shared fakes: an in-process exchange gateway and an info client with scripted account state."""
from __future__ import annotations


class FakeInfo:
    def __init__(self, mids, positions=None, equity="5000.0", fills=None, open_orders=None):
        self.mids, self.positions, self.equity = dict(mids), list(positions or []), equity
        self.fills, self.orders = list(fills or []), list(open_orders or [])
        self.calls = []

    async def all_mids(self):
        self.calls.append("allMids")
        return {k: float(v) for k, v in self.mids.items()}     # the real client returns floats

    async def user_state(self, address):
        self.calls.append("clearinghouseState")
        return {"marginSummary": {"accountValue": str(self.equity)},
                "assetPositions": [{"type": "oneWay", "position": {"coin": c, "szi": str(s)}} for c, s in self.positions]}

    async def user_fills_by_time(self, address, start_ms):
        self.calls.append("userFillsByTime")
        return [f for f in self.fills if f["time"] >= start_ms]

    async def open_orders(self, address):
        self.calls.append("openOrders")
        return list(self.orders)


class FakeGateway:
    def __init__(self, fill_frac=1.0, fail_entry=None, fail_stop=False):
        self.calls, self.fill_frac, self.fail_entry, self.fail_stop = [], fill_frac, fail_entry, fail_stop
        self.oid = 100

    def update_leverage(self, leverage, coin, is_cross=True):
        self.calls.append(("lev", coin, leverage, is_cross))
        return {"status": "ok", "response": {"type": "default"}}

    def cancel(self, coin, oid):
        self.calls.append(("cancel", coin, oid))
        return {"status": "ok", "response": {"type": "cancel", "data": {"statuses": ["success"]}}}

    def order(self, coin, is_buy, sz, limit_px, order_type, reduce_only=False):
        self.calls.append(("order", coin, is_buy, sz, limit_px, order_type, reduce_only))
        self.oid += 1
        if "trigger" in order_type:
            if self.fail_stop:
                return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": "Reduce only order would increase position."}]}}}
            return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": self.oid}}]}}}
        if self.fail_entry:
            return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": self.fail_entry}]}}}
        filled = round(sz * self.fill_frac, 1)
        avg = limit_px - 0.01 if is_buy else limit_px + 0.01
        return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"totalSz": str(filled), "avgPx": str(avg), "oid": self.oid}}]}}}
