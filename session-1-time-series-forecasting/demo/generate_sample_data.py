"""
=============================================================================
  DEMO: Generate Sample Energy Demand Data
  Session 1 — Helper Script
=============================================================================

  Run this script BEFORE the main demo to generate sample data.
  It creates a realistic energy demand dataset with:
    - Hourly timestamps
    - Demand (target variable)
    - Temperature, humidity (features)
    - Holiday flag

  Usage:
    python generate_sample_data.py
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuration
NUM_DAYS = 365  # 1 year of hourly data
START_DATE = "2025-01-01"
OUTPUT_DIR = Path(__file__).parent / "train_data"
OUTPUT_FILE = OUTPUT_DIR / "train.csv"

np.random.seed(42)

# Generate hourly timestamps
dates = pd.date_range(start=START_DATE, periods=NUM_DAYS * 24, freq="h")

# Base demand pattern
# - Daily cycle: higher during work hours, lower at night
# - Weekly cycle: lower on weekends
# - Seasonal: higher in summer/winter (HVAC)

hours = dates.hour
days_of_week = dates.dayofweek
day_of_year = dates.dayofyear

# Daily pattern (peaks at 9am and 7pm)
daily_pattern = (
    30 * np.sin(2 * np.pi * (hours - 6) / 24)
    + 15 * np.sin(2 * np.pi * (hours - 18) / 12)
)

# Weekly pattern (10% lower on weekends)
weekend_factor = np.where(days_of_week >= 5, 0.9, 1.0)

# Seasonal pattern (peaks in summer and winter)
seasonal_pattern = 50 * np.cos(2 * np.pi * (day_of_year - 200) / 365)

# Base demand
base_demand = 500

# Combine
demand = (
    base_demand
    + daily_pattern * weekend_factor
    + seasonal_pattern
    + np.random.normal(0, 15, len(dates))  # noise
)
demand = np.maximum(demand, 100)  # floor at 100

# Temperature (correlated with season)
temperature = (
    20
    + 15 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    + 5 * np.sin(2 * np.pi * hours / 24)
    + np.random.normal(0, 3, len(dates))
)

# Humidity
humidity = (
    60
    + 20 * np.sin(2 * np.pi * (day_of_year - 200) / 365)
    + np.random.normal(0, 8, len(dates))
)
humidity = np.clip(humidity, 10, 100)

# Holiday flag (US federal holidays approximate)
us_holidays = pd.to_datetime([
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13",
    "2025-11-11", "2025-11-27", "2025-12-25",
])
is_holiday = dates.normalize().isin(us_holidays).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "timestamp": dates,
    "demand": np.round(demand, 2),
    "temperature": np.round(temperature, 2),
    "humidity": np.round(humidity, 2),
    "is_holiday": is_holiday,
})

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Generated {len(df)} rows of sample energy demand data.")
print(f"📁 Saved to: {OUTPUT_FILE}")
print(f"\nSample:\n{df.head(10)}")
print(f"\nStatistics:\n{df.describe()}")

# Create MLTable definition
mltable_content = """$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json
type: mltable

paths:
  - file: ./train.csv

transformations:
  - read_delimited:
      delimiter: ","
      header: all_files_same_headers
      encoding: utf8
"""

mltable_path = OUTPUT_DIR / "MLTable"
with open(mltable_path, "w") as f:
    f.write(mltable_content)

print(f"\n✅ MLTable definition created at: {mltable_path}")
print("\n🎉 Sample data is ready for the forecasting demo!")
