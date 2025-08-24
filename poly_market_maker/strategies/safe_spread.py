import logging
from typing import List, Optional, Dict

from poly_market_maker.constants import MIN_TICK, MIN_SIZE, MAX_DECIMALS
from poly_market_maker.order import Order, Side
from poly_market_maker.token import Token


def _round_price(x: float) -> float:
    return round(max(MIN_TICK, min(1.0 - MIN_TICK, x)), MAX_DECIMALS)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SafeSpread:
    """
    Inventory-capped, symmetric maker that *for each buy_token loop*:
      - quotes a BID on buy_token
      - quotes an ASK on buy_token.complement()

    It cancels anything far from the current target, limits number of quotes,
    skews away from inventory, and respects simple per-token & notional caps.
    """

    def __init__(self, cfg: dict):
        self.logger = logging.getLogger(self.__class__.__name__)

        def get(keys, default):
            for k in keys:
                if k in cfg:
                    return cfg[k]
            return default

        self.target_spread = float(get(["target_spread", "targetSpread"], 0.03))
        self.half_spread_override = get(["half_spread_override", "halfSpreadOverride"], None)
        self.order_size = float(get(["order_size", "orderSize"], 10.0))
        self.max_bids = int(get(["max_bids", "maxBids"], 2))
        self.max_asks = int(get(["max_asks", "maxAsks"], 2))
        self.cancel_if_away_by = float(get(["cancel_if_away_by", "cancelIfAwayBy"], 0.02))
        self.max_inventory_per_token = float(get(["max_inventory_per_token", "maxInventoryPerToken"], 40.0))
        self.max_notional_per_side = float(get(["max_notional_per_side", "maxNotionalPerSide"], 120.0))
        self.skew_per_unit = float(get(["skew_per_unit", "skewPerUnit"], 0.0005))
        self.min_place_size = float(get(["min_place_size", "minPlaceSize"], 0.1))

        # live context (fed by the Strategy wrapper each cycle)
        self._balances: Dict[Token, float] = {t: 0.0 for t in Token}
        self._free_collateral: float = 0.0

    # ---------- context from wrapper ----------

    def set_context(self, balances: Dict[Token, float], free_collateral: float):
        self._balances = dict(balances)
        self._free_collateral = float(free_collateral)

    # ---------- math helpers ----------

    def _mid_with_skew(self, base_mid: float, token: Token) -> float:
        inv = float(self._balances.get(token, 0.0))
        skew = inv * self.skew_per_unit  # long -> shift mid down (discourage more longs)
        return _round_price(base_mid - skew)

    def _desired_quotes(self, mid: float) -> tuple[float, float]:
        half = self.half_spread_override if self.half_spread_override is not None else (self.target_spread / 2.0)
        half = max(half, MIN_TICK)
        bid = _round_price(mid - half)
        ask = _round_price(mid + half)
        if ask <= bid:
            ask = _round_price(bid + MIN_TICK)
        return bid, ask

    # ---------- cancellation ----------

    def cancellable_orders(self, orders: List[Order], target_price: float) -> List[Order]:
        """
        `orders` contains: BUYs on buy_token and SELLs on buy_token.complement().
        Cancel those that are far from targets and trim to {max_bids,max_asks}.
        """
        if not orders:
            return []

        # deduce buy_token from any BUY; else fall back to SELL's complement
        buy_token: Optional[Token] = None
        for o in orders:
            if o.side == Side.BUY:
                buy_token = o.token
                break
        if buy_token is None:
            # no BUYs? then any SELL must be on complement
            for o in orders:
                if o.side == Side.SELL:
                    buy_token = o.token.complement()
                    break
        if buy_token is None:
            # nothing to do
            return []

        mid = self._mid_with_skew(target_price, buy_token)
        bid_target, ask_target = self._desired_quotes(mid)

        def effective_price(order: Order) -> float:
            # SELL on complement is expressed in buy_token price space as (1 - price)
            return order.price if order.side == Side.BUY else round(1.0 - order.price, MAX_DECIMALS)

        # 1) cancel far-away quotes
        far_aways = [o for o in orders if abs(effective_price(o) - (bid_target if o.side == Side.BUY else ask_target)) > self.cancel_if_away_by]

        # 2) limit counts, keep closest to target
        def closeness(o: Order) -> float:
            return abs(effective_price(o) - (bid_target if o.side == Side.BUY else ask_target))

        keep_buys = [o for o in orders if o.side == Side.BUY and o not in far_aways]
        keep_sells = [o for o in orders if o.side == Side.SELL and o not in far_aways]

        keep_buys.sort(key=closeness)
        keep_sells.sort(key=closeness)

        cancel_extra_buys = keep_buys[self.max_bids:] if len(keep_buys) > self.max_bids else []
        cancel_extra_sells = keep_sells[self.max_asks:] if len(keep_sells) > self.max_asks else []

        return far_aways + cancel_extra_buys + cancel_extra_sells

    # ---------- placement ----------

    def new_orders(
            self,
            orders: List[Order],
            collateral_balance: float,
            token_balance: float,   # NOTE: this is *complement token* balance (inventory available to sell)
            target_price: float,
            buy_token: Token,
    ) -> List[Order]:
        new_orders: List[Order] = []

        # current desired quotes (in buy_token price space)
        mid = self._mid_with_skew(target_price, buy_token)
        bid_target, ask_target = self._desired_quotes(mid)

        # check if we already have near-target quotes (avoid spamming nearly-duplicates)
        def has_near(side: Side, price_buy_space: float, eps: float = MIN_TICK) -> bool:
            for o in orders:
                if o.side != side:
                    continue
                p_eff = o.price if side == Side.BUY else round(1.0 - o.price, MAX_DECIMALS)
                if abs(p_eff - price_buy_space) <= eps:
                    return True
            return False

        # ----- ASK on complement token -----
        # Map desired ask in buy_token space -> actual sell price on complement:
        sell_token = buy_token.complement()
        sell_price_actual = _round_price(1.0 - ask_target)

        if (not has_near(Side.SELL, ask_target)) and self._can_place_ask(token_balance):
            ask_size = self._ask_size(token_balance, sell_token)
            if ask_size >= max(self.min_place_size, MIN_SIZE):
                new_orders.append(Order(price=sell_price_actual, size=ask_size, side=Side.SELL, token=sell_token))

        # ----- BID on buy_token -----
        if (not has_near(Side.BUY, bid_target)) and self._can_place_bid(collateral_balance, bid_target, buy_token):
            bid_size = self._bid_size(collateral_balance, bid_target, buy_token)
            if bid_size >= max(self.min_place_size, MIN_SIZE):
                new_orders.append(Order(price=bid_target, size=bid_size, side=Side.BUY, token=buy_token))

        return new_orders

    # ---------- caps & sizing ----------

    def _can_place_ask(self, token_balance: float) -> bool:
        return token_balance >= max(self.min_place_size, MIN_SIZE)

    def _ask_size(self, token_balance: float, token: Token) -> float:
        # Standard size, but if we're above cap on this token, sell faster.
        size = min(self.order_size, token_balance)
        inv = float(self._balances.get(token, 0.0))
        if inv > self.max_inventory_per_token:
            over = inv - self.max_inventory_per_token
            size = max(size, min(over, token_balance))
        return round(_clamp(size, MIN_SIZE, token_balance), MAX_DECIMALS)

    def _can_place_bid(self, collateral_balance: float, price: float, buy_token: Token) -> bool:
        if collateral_balance < price * max(self.min_place_size, MIN_SIZE):
            return False
        if (price * self.order_size) > self.max_notional_per_side:
            return False
        # respect inventory cap for the buy_token (approx.: current holdings only)
        inv_now = float(self._balances.get(buy_token, 0.0))
        if inv_now >= self.max_inventory_per_token:
            return False
        return True

    def _bid_size(self, collateral_balance: float, price: float, buy_token: Token) -> float:
        max_by_cash = collateral_balance / price
        max_by_notional = self.max_notional_per_side / price
        inv_now = float(self._balances.get(buy_token, 0.0))
        headroom = max(0.0, self.max_inventory_per_token - inv_now)
        size = min(self.order_size, max_by_cash, max_by_notional, headroom)
        return round(_clamp(size, MIN_SIZE, 1e9), MAX_DECIMALS)
