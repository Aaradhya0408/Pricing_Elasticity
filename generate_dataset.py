"""Generate a synthetic e-commerce sales dataset with KNOWN price elasticities.

Because we plant the true elasticity per category, the analysis notebook can be
checked: the regression should recover values close to the truth.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# category -> (base price, true elasticity, base daily units, n products)
CATS = {
    "Electronics": (250.0, -1.6, 14, 6),
    "Fashion":     (45.0,  -2.2, 40, 8),
    "Home":        (80.0,  -1.3, 22, 6),
    "Beauty":      (25.0,  -1.8, 55, 6),
    "Grocery":     (8.0,   -0.7, 120, 6),
}
CHANNELS = ["web", "mobile_app", "marketplace"]
CH_MULT = {"web": 1.0, "mobile_app": 1.15, "marketplace": 0.85}
dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

rows, pid = [], 100
for cat, (bp, eps, bu, n) in CATS.items():
    for _ in range(n):
        pid += 1
        p_scale = rng.uniform(0.7, 1.4)           # product-level price level
        u_scale = rng.uniform(0.7, 1.3)           # product-level popularity
        ref_price = bp * p_scale
        for d in dates:
            ch = rng.choice(CHANNELS, p=[0.45, 0.35, 0.20])
            comp = ref_price * rng.normal(1.0, 0.05)
            promo = int(rng.random() < 0.12)
            # seller prices around competitor, with promo discounts and noise
            price = comp * rng.normal(1.0, 0.08) * (0.88 if promo else 1.0)
            season = 1 + 0.25 * np.sin(2 * np.pi * (d.dayofyear - 300) / 365)
            if d.month in (11, 12):
                season *= 1.2                       # festive uplift
            weekend = 1.1 if d.dayofweek >= 5 else 1.0
            mu = (bu * u_scale * season * weekend * CH_MULT[ch]
                  * (price / ref_price) ** eps
                  * (1.10 if promo else 1.0)        # promo visibility lift
                  * (comp / ref_price) ** 0.3)       # cross-price effect (competitor)
            units = rng.poisson(max(mu, 0.1))
            rows.append((d.date(), f"P{pid}", cat, round(price, 2), int(units),
                         round(comp, 2), promo, ch))

df = pd.DataFrame(rows, columns=["date", "product_id", "category", "price",
                                 "units_sold", "competitor_price",
                                 "promotion_flag", "channel"])

# Inject realistic mess so the cleaning step has real work to do
df = df.sample(frac=1, random_state=1).reset_index(drop=True)
idx = rng.choice(len(df), 40, replace=False)
df.loc[idx[:15], "price"] = np.nan
df.loc[idx[15:30], "competitor_price"] = np.nan
df.loc[idx[30:35], "units_sold"] = -1                  # invalid
df.loc[idx[35:], "category"] = df.loc[idx[35:], "category"].str.lower()
df = pd.concat([df, df.sample(25, random_state=3)], ignore_index=True)  # dupes

df.to_csv("dataset.csv", index=False)

dd = pd.DataFrame([
    ("date", "date (YYYY-MM-DD)", "Day of the sale record"),
    ("product_id", "string", "Unique product identifier (P101...)"),
    ("category", "string", "Product category: Electronics, Fashion, Home, Beauty, Grocery"),
    ("price", "float", "Selling price per unit charged that day (currency units)"),
    ("units_sold", "int", "Units sold of this product on this day via this channel"),
    ("competitor_price", "float", "Average competitor price for the equivalent product that day"),
    ("promotion_flag", "int (0/1)", "1 if a promotion/discount campaign was running"),
    ("channel", "string", "Sales channel: web, mobile_app, marketplace"),
], columns=["column", "type", "description"])
dd.to_csv("data_dictionary.csv", index=False)
print(df.shape); print(df.head()); print(df.isna().sum())
print({k: v[1] for k, v in CATS.items()})
