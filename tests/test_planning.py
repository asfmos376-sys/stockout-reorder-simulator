from pathlib import Path
import unittest

import pandas as pd

from planning import MODELS, evaluate, forecast, reorder_point, simulate


class PlanningTests(unittest.TestCase):
    def setUp(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "demand_history.csv"
        observations = pd.read_csv(data_path, parse_dates=["date"])
        self.history = observations.loc[
            observations["sku"] == "WATER-001"
        ].set_index("date")["units"]
        self.demand = forecast(self.history, MODELS[0], 28)

    def test_default_demo(self):
        for lead, threshold, lost in [(7, 140, 40), (10, 200, 100)]:
            with self.subTest(lead=lead):
                plan = simulate(self.demand, 100, lead, 500)
                self.assertEqual(reorder_point(self.demand, lead, 0), threshold)
                self.assertEqual(plan["Unmet demand"].sum(), lost)
                self.assertEqual((plan["Unmet demand"] > 0).sum(), lead - 5)
                self.assertEqual(plan["Received"].iloc[lead], 500)
                self.assertEqual(plan["Received"].sum(), 500)

    def test_receipt_before_sales_and_buffer(self):
        plan = simulate(self.demand, 140, 7, 1000)
        self.assertEqual(plan["Unmet demand"].sum(), 0)
        self.assertEqual(reorder_point(self.demand, 7, 10), 150)

    def test_seasonal_forecast_and_backtest(self):
        weekly = pd.Series(list(range(1, 8)) * 12, index=self.history.index)
        scores = evaluate(weekly, 10)
        self.assertEqual(scores.index[0], MODELS[1])
        self.assertEqual(scores.iloc[0], 0)
        self.assertEqual(forecast(weekly, MODELS[1], 9).tolist(),
                         [1, 2, 3, 4, 5, 6, 7, 1, 2])


if __name__ == "__main__":
    unittest.main()
