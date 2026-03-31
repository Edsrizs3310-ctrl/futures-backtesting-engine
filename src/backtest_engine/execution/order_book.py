"""
Single-engine resting order registry.

This module intentionally stays small and deterministic. It is not a full OMS
for the entire repository yet; it provides the single-asset engine with an
explicit place to own order state transitions and active resting orders.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable, List, Optional
from uuid import uuid4

from . import Fill, Order


class OrderBook:
    """
    Maintains active single-engine orders across bars.

    Methodology:
        Orders are submitted once, then carried bar-to-bar until they reach a
        terminal state. The engine remains responsible for session policy and
        liquidation priority; the book owns storage and state transitions.
    """

    def __init__(self) -> None:
        self._active_orders: List[Order] = []

    def has_open_orders(self) -> bool:
        """Returns True when any active order remains in the registry."""
        return bool(self._active_orders)

    def active_orders(self) -> List[Order]:
        """Returns a shallow copy of the active order list."""
        return list(self._active_orders)

    def submit(self, order: Order, placed_at) -> None:
        """
        Submits a single order into the active registry.
        """
        if order.timestamp is None:
            order.timestamp = placed_at
        if order.placed_at is None:
            order.placed_at = placed_at
        if order.status == "NEW":
            order.status = "SUBMITTED"
        self._active_orders.append(order)

    def submit_many(self, orders: List[Order], placed_at) -> None:
        """
        Submits multiple orders at the same engine timestamp.
        """
        self._assign_protective_oco_metadata(orders)
        for order in orders:
            self.submit(order, placed_at)

    def cancel(self, order: Order) -> None:
        """
        Cancels an order and removes it from the active registry.
        """
        order.status = "CANCELLED"
        self._active_orders = [active for active in self._active_orders if active.id != order.id]

    def cancel_where(self, predicate: Callable[[Order], bool]) -> List[Order]:
        """
        Cancels every active order that matches the predicate.
        """
        cancelled: List[Order] = []
        kept: List[Order] = []
        for order in self._active_orders:
            if predicate(order):
                order.status = "CANCELLED"
                cancelled.append(order)
            else:
                kept.append(order)
        self._active_orders = kept
        return cancelled

    def pull_where(self, predicate: Callable[[Order], bool]) -> List[Order]:
        """
        Removes matching active orders without changing their status.
        """
        pulled: List[Order] = []
        kept: List[Order] = []
        for order in self._active_orders:
            if predicate(order):
                pulled.append(order)
            else:
                kept.append(order)
        self._active_orders = kept
        return pulled

    def cancel_expired_day_orders(self, current_date: date) -> List[Order]:
        """
        Cancels DAY orders once the engine crosses into a later calendar date.
        """
        return self.cancel_where(
            lambda order: (
                str(order.time_in_force).upper() == "DAY"
                and order.placed_at is not None
                and self._placement_date(order) < current_date
            )
        )

    def process_active_orders(
        self,
        attempt_fill: Callable[[Order], Optional[Fill]],
        can_attempt: Callable[[Order], bool],
        preview_fill: Optional[Callable[[Order], Optional[float]]] = None,
        select_oco_winner: Optional[Callable[[List[Order]], Order]] = None,
    ) -> List[Fill]:
        """
        Attempts fills for all active orders and retains non-terminal residue.
        """
        retained: List[Order] = []
        fills: List[Fill] = []

        for group in self._group_active_orders():
            if len(group) == 1 or group[0].oco_group_id is None or preview_fill is None:
                self._process_non_oco_group(
                    group=group,
                    attempt_fill=attempt_fill,
                    can_attempt=can_attempt,
                    retained=retained,
                    fills=fills,
                )
                continue

            eligible = [order for order in group if can_attempt(order)]
            blocked = [order for order in group if not can_attempt(order)]
            fillable = [
                order
                for order in eligible
                if preview_fill(order) is not None
            ]

            if not fillable:
                self._process_non_oco_group(
                    group=eligible,
                    attempt_fill=attempt_fill,
                    can_attempt=lambda _order: True,
                    retained=retained,
                    fills=fills,
                )
                retained.extend(
                    order
                    for order in blocked
                    if order.status not in {"CANCELLED", "REJECTED", "FILLED"}
                )
                continue

            winner = (
                select_oco_winner(fillable)
                if select_oco_winner is not None
                else self._select_oco_winner(fillable)
            )
            fill = attempt_fill(winner)
            if fill is not None:
                fills.append(fill)
                for sibling in group:
                    if sibling.id == winner.id:
                        continue
                    sibling.status = "CANCELLED"
                continue

            for order in group:
                if order.status not in {"CANCELLED", "REJECTED", "FILLED"}:
                    retained.append(order)

        self._active_orders = retained
        return fills

    @staticmethod
    def _assign_protective_oco_metadata(orders: List[Order]) -> None:
        """
        Auto-tags same-bar protective brackets into one OCO group.

        Methodology:
            Legacy single-asset strategies still emit a flat List[Order] without
            an explicit bracket object. To keep that API unchanged while adding
            safe native STOP/LIMIT exits, multiple same-bar reduce-only
            non-market orders are treated as one protective bracket candidate,
            mirroring the existing portfolio-engine bridge semantics.
        """
        protective_orders = [
            order
            for order in orders
            if bool(order.reduce_only) and str(order.order_type).upper() != "MARKET"
        ]
        if len(protective_orders) < 2:
            return

        group_id = uuid4().hex
        for order in protective_orders:
            if order.oco_group_id is None:
                order.oco_group_id = group_id
            if order.oco_role is None:
                order.oco_role = OrderBook._infer_oco_role(order)

    @staticmethod
    def _infer_oco_role(order: Order) -> str:
        """
        Returns the coarse protective role used by same-bar OCO resolution.
        """
        order_type = str(order.order_type).upper()
        if order_type in {"STOP", "STOP_LIMIT"}:
            return "STOP"
        return "TARGET"

    @staticmethod
    def _select_oco_winner(orders: List[Order]) -> Order:
        """
        Picks the deterministic fill winner inside an OCO group.

        Methodology:
            On a coarse OHLC bar we cannot observe the true intrabar path. The
            single-engine fallback is therefore pessimistic: if both stop and
            target are reachable on the same bar, the stop wins.
        """
        stops = [
            order
            for order in orders
            if str(order.oco_role or OrderBook._infer_oco_role(order)).upper() == "STOP"
        ]
        if stops:
            return sorted(stops, key=lambda order: order.id)[0]
        return sorted(orders, key=lambda order: order.id)[0]

    @staticmethod
    def _process_non_oco_group(
        group: List[Order],
        attempt_fill: Callable[[Order], Optional[Fill]],
        can_attempt: Callable[[Order], bool],
        retained: List[Order],
        fills: List[Fill],
    ) -> None:
        """
        Applies the legacy independent-order processing path to one group.
        """
        for order in group:
            if not can_attempt(order):
                retained.append(order)
                continue

            fill = attempt_fill(order)
            if fill is not None:
                fills.append(fill)
                continue

            if order.status not in {"CANCELLED", "REJECTED", "FILLED"}:
                retained.append(order)

    def _group_active_orders(self) -> List[List[Order]]:
        """
        Returns active orders grouped by OCO group identifier.
        """
        grouped: List[List[Order]] = []
        seen: set[str] = set()
        for order in self._active_orders:
            group_id = order.oco_group_id or order.id
            if group_id in seen:
                continue
            seen.add(group_id)
            grouped.append(
                [
                    active
                    for active in self._active_orders
                    if (active.oco_group_id or active.id) == group_id
                ]
            )
        return grouped

    @staticmethod
    def _placement_date(order: Order) -> date:
        """
        Normalizes placement timestamps into calendar dates.
        """
        if isinstance(order.placed_at, datetime):
            return order.placed_at.date()
        if isinstance(order.placed_at, date):
            return order.placed_at
        raise TypeError("Order.placed_at must be date-like when expiring DAY orders.")
