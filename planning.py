"""Demand forecasting and daily inventory simulation."""
import math

import pandas as pd

MODELS = ("7-day moving average", "Weekly seasonal naive")


def evaluate(history, horizon):
    """Score fixed-origin forecasts at the maximum supplier lead time."""
    window = len(history) - 28
    errors = {}
    for model in MODELS:
        absolute_errors = []
        for origin in range(window, len(history)):
            actual = history.iloc[origin:origin + horizon]
            predicted = forecast(history.iloc[:origin], model, len(actual))
            absolute_errors.extend((actual - predicted).abs())
        errors[model] = sum(absolute_errors) / len(absolute_errors)
    return pd.Series(errors, name="MAE (units)").sort_values(kind="stable")


def forecast(history, model, days):
    recent = history.iloc[-7:]
    dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=days)
    if model == MODELS[0]:
        values = [math.floor(recent.mean() + 0.5)] * days
    elif model == MODELS[1]:
        values = [int(recent.iloc[day % 7]) for day in range(days)]
    else:
        raise ValueError(f"Unknown model: {model}")
    return pd.Series(values, index=dates, name="Forecast demand", dtype=int)


def reorder_point(demand, lead_time, safety_stock):
    """Cover demand over the full days before receipt, plus buffer."""
    return int(demand.iloc[:lead_time].sum()) + safety_stock


def simulate(demand, stock, lead_time, order_quantity):
    """Receive one order before sales on day lead_time + 1."""
    rows = []
    for offset, (date, units) in enumerate(demand.items()):
        received = order_quantity if offset == lead_time else 0
        available = stock + received
        sold = min(available, units)
        stock = available - sold
        rows.append((date, available - received, units, received, sold, units - sold, stock))
    return pd.DataFrame(
        rows,
        columns=["Date", "Opening stock", "Forecast demand", "Received", "Sold",
                 "Unmet demand", "Closing stock"],
    ).set_index("Date")


def summarize(plan):
    shortages = plan.loc[plan["Unmet demand"] > 0, "Unmet demand"]
    return {
        "First shortage date": shortages.index[0].date().isoformat() if len(shortages) else "None",
        "Days with unmet demand": str(len(shortages)),
        "Units of unmet demand": str(shortages.sum()),
    }
