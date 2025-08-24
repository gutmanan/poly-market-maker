import logging
from typing import List, Dict

from poly_market_maker.token import Token, Collateral
from poly_market_maker.order import Order, Side
from poly_market_maker.orderbook import OrderBook
from poly_market_maker.strategies.base_strategy import BaseStrategy
from poly_market_maker.strategies.safe_spread import SafeSpread


class SafeSpreadStrategy(BaseStrategy):
    def __init__(self, config: dict):
        assert isinstance(config, dict)
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        # Accept either {"safe_spread": {...}} or a flat dict of keys
        cfg = config.get("safe_spread", config)
        self.mm = SafeSpread(cfg)

    def get_orders(self, orderbook: OrderBook, target_prices):
        """
        1) Cancel off-center/extra quotes.
        2) Compute free collateral & free token balances.
        3) Place at most one near-target BID (buy_token) and one ASK (complement) per loop.
        """
        orders_to_cancel: List[Order] = []
        orders_to_place: List[Order] = []

        # ---- log prices
        for token in Token:
            self.logger.debug(f"{token.value} target price: {target_prices[token]}")

        # ---- gather per-token current 'corresponding' orders
        by_token_orders: Dict[Token, List[Order]] = {}
        for token in Token:
            by_token_orders[token] = self._orders_by_corresponding_buy_token(orderbook.orders, token)

        # ---- cancel phase
        for token in Token:
            orders_to_cancel += self.mm.cancellable_orders(by_token_orders[token], target_prices[token])

        # remaining open orders after cancellation
        open_orders = [o for o in orderbook.orders if o not in set(orders_to_cancel)]

        # ---- collateral locked by remaining open BUYS (all tokens)
        balance_locked_by_open_buys = sum(o.size * o.price for o in open_orders if o.side == Side.BUY)
        free_collateral_balance = orderbook.balances[Collateral] - balance_locked_by_open_buys
        self.logger.debug(f"Free collateral balance: {free_collateral_balance}")

        # ---- compute free balances per token (inventory not locked by SELLs)
        locked_sells_by_token: Dict[Token, float] = {t: 0.0 for t in Token}
        for o in open_orders:
            if o.side == Side.SELL:
                locked_sells_by_token[o.token] += o.size

        free_token_balance_by_token: Dict[Token, float] = {
            t: orderbook.balances[t] - locked_sells_by_token[t] for t in Token
        }

        # also pass *total* balances for inventory-capping logic
        self.mm.set_context(balances={t: orderbook.balances[t] for t in Token},
                            free_collateral=free_collateral_balance)

        # ---- placement phase
        for token in Token:
            # In this loop, SELLs will be placed on token.complement()
            free_sell_token_balance = free_token_balance_by_token[token.complement()]
            new_orders = self.mm.new_orders(
                by_token_orders[token],
                free_collateral_balance,
                free_sell_token_balance,
                target_prices[token],
                token,
            )
            # update free collateral by the newly placed BUYS
            free_collateral_balance -= sum(o.size * o.price for o in new_orders if o.side == Side.BUY)
            orders_to_place += new_orders

        return (orders_to_cancel, orders_to_place)

    # ----- helpers identical to BandsStrategy -----

    def _orders_by_corresponding_buy_token(self, orders: List[Order], buy_token: Token) -> List[Order]:
        return [o for o in orders if self._filter_by_corresponding_buy_token(o, buy_token)]

    def _filter_by_corresponding_buy_token(self, order: Order, buy_token: Token) -> bool:
        # BUYs on buy_token, SELLs on complement
        return (order.side == Side.BUY and order.token == buy_token) or (order.side == Side.SELL and order.token != buy_token)
