from pathlib import Path

import pandas as pd
import streamlit as st

from planning import evaluate, forecast, reorder_point, simulate, summarize

DATA = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="Stockout & Reorder Simulator", layout="wide")


@st.cache_data
def load_data():
    products = pd.read_csv(DATA / "products.csv").set_index("product")
    demand = pd.read_csv(DATA / "demand_history.csv", parse_dates=["date"])
    return products, demand


products, observations = load_data()
st.title("Stockout & Reorder Simulator")
st.caption("Inventory planning demo using synthetic daily demand.")

with st.sidebar:
    st.header("Scenario inputs")
    selected = st.selectbox("Product", products.index)
    product = products.loc[selected]
    sku = product["sku"]
    stock = st.number_input("Starting stock (units)", min_value=0, step=1,
                            value=int(product["starting_stock"]), key=f"{sku}-stock")
    lead = st.slider("Normal supplier lead time (days)", 1, 21,
                     int(product["lead_time"]), key=f"{sku}-lead")
    delay = st.slider("Extra supplier delay (days)", 0, 14, 3)
    quantity = st.number_input("Order quantity (units)", min_value=1, step=1,
                               value=int(product["order_quantity"]), key=f"{sku}-qty")
    safety = st.number_input("Safety stock (units)", min_value=0, step=1, value=0)

history = observations.loc[observations["sku"] == sku].set_index("date")["units"].sort_index()
delayed_lead = lead + delay
scores = evaluate(history, delayed_lead)
model = scores.index[0]
demand = forecast(history, model, max(28, delayed_lead + 7))

normal_threshold = reorder_point(demand, lead, safety)
delayed_threshold = reorder_point(demand, delayed_lead, safety)
normal = simulate(demand, stock, lead, quantity)
delayed = simulate(demand, stock, delayed_lead, quantity)
normal_label, delayed_label = f"Normal ({lead} days)", f"Delayed ({delayed_lead} days)"

if stock <= normal_threshold:
    st.warning("Reorder now under the normal lead-time assumption.")
elif stock <= delayed_threshold:
    st.warning("Reorder now if the supplier delay is expected.")
else:
    st.success("Stock is above both reorder points. Continue monitoring.")
st.caption("The simulation below assumes you place the order on the first forecast date.")

first, second, third = st.columns(3)
first.metric("Current stock", f"{stock:,} units")
second.metric("Normal reorder point", f"{normal_threshold:,} units")
third.metric("Delayed reorder point", f"{delayed_threshold:,} units")

inventory_tab, forecast_tab, export_tab = st.tabs(["Inventory", "Demand forecast", "Daily plan"])
with inventory_tab:
    st.subheader("Closing inventory by scenario")
    st.line_chart(pd.DataFrame({
        normal_label: normal["Closing stock"],
        delayed_label: delayed["Closing stock"],
    }))
    st.table(pd.DataFrame({
        normal_label: summarize(normal),
        delayed_label: summarize(delayed),
    }))
    st.caption("A shortage day has unmet demand. Unmet units are counted as lost sales.")

with forecast_tab:
    st.subheader("Daily demand and forecast")
    st.line_chart(pd.concat([
        history.tail(28).rename("Observed demand"),
        demand,
    ], axis=1))
    st.write(f"Selected model: **{model}**")
    st.table(scores.round(2).to_frame())
    st.caption("Rolling backtest over the last 28 observations. Each origin forecasts up "
               "to the delayed lead time, limited by the available observations.")

with export_tab:
    scenario = st.radio("Scenario", ["Normal", "Delayed"], horizontal=True)
    plan = normal if scenario == "Normal" else delayed
    st.dataframe(plan)
    st.download_button("Download daily plan", plan.to_csv(),
                       f"{sku}-{scenario.lower()}-plan.csv", "text/csv")
    st.download_button("Download demand history", history.rename("units").to_csv(),
                       f"{sku}-history.csv", "text/csv")
