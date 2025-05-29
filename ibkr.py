from ib_insync import IB, Contract, ComboLeg, Order


def send_combo_order_from_legs(
    legs: list[dict],
    symbol: str,
    currency: str,
    exchange: str,
    total_quantity: int,
    order_type: str,  # 'LMT' or 'MKT'
    lmt_price: float | None,
    client_id: int = 0,
    host: str = "127.0.0.1",
    port: int = 4001,
    ib=None,
):
    if ib is None:
        ib = IB()
        ib.connect(host, port, clientId=client_id)

    combo = Contract()
    combo.symbol = symbol
    combo.secType = "BAG"
    combo.currency = currency
    combo.exchange = exchange
    combo.comboLegs = []

    for leg in legs:
        cl = ComboLeg()
        cl.ratio = 1
        cl.action = leg["action"]
        cl.exchange = leg["exchange"]
        cl.conId = leg["conId"]
        cl.openClose = 0
        combo.comboLegs.append(cl)

    order = Order()
    order.action = "BUY"
    order.orderType = order_type
    order.totalQuantity = total_quantity

    if order_type == "LMT":
        if lmt_price is None:
            raise ValueError("Must supply lmt_price for LMT orders")
        order.lmtPrice = lmt_price

    trade = ib.placeOrder(combo, order)
    ib.sleep(1)
    return trade
