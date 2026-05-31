import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Define color mapping
energyMixColorMapping = {
    'wind_onshore': 'blue',
    'wind_offshore': 'cyan',
    'solar': '#FFD700',
    'gas': '#808080',
    'hard_coal': '#000000',
    'lignite': '#8B4513',
    'renewables': '#008000',
}

# Load JSON data
file_path = 'data/DE/forecasts/energy_mix/forecast_prev_actual.json'
with open(file_path, 'r') as file:
    energy_data = json.load(file)

# Process data
energy_sources = {}
time_stamps = set()

for source in energy_data:
    name = source['name']
    values = source['data']
    energy_sources[name] = {}
    for timestamp, value in values:
        energy_sources[name][timestamp] = value
        time_stamps.add(timestamp)

# Convert to DataFrame
time_stamps = sorted(time_stamps, key=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M"))
df = pd.DataFrame(index=time_stamps)

for source, values in energy_sources.items():
    df[source] = [values.get(ts, 0) for ts in time_stamps]

# Convert index to datetime
df.index = pd.to_datetime(df.index)

# Plot stacked area chart
plt.figure(figsize=(12, 6))
plt.stackplot(df.index, df.T, labels=df.columns, colors=[energyMixColorMapping.get(col, 'gray') for col in df.columns])

plt.legend(loc='upper left')
plt.xlabel("Time")
plt.ylabel("Energy Output (MW)")
plt.title("Stacked Area Plot of Energy Mix Forecast")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# plt.plot(df.index, df['wind_onshore'], color='blue')
# plt.plot(df.index, df['wind_offshore'], color='cyan')
# plt.plot(df.index, df['solar'], color='gold')

# plt.show()