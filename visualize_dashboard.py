import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load vehicle log CSV
csv_path = 'results/vehicle_log.csv'
if not os.path.exists(csv_path):
    print("CSV log not found. Run detect_vehicles.py first.")
    exit()

df = pd.read_csv(csv_path)

# Convert time if needed
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S', errors='coerce')

# Basic stats
total_vehicles = df['Vehicle ID'].nunique()
avg_speed = df['Speed (km/h)'].mean()

print(f"Total Vehicles: {total_vehicles}")
print(f"Average Speed: {avg_speed:.2f} km/h")

# Plot 1: Speed vs Vehicle ID
plt.figure(figsize=(10, 5))
sns.boxplot(x='Vehicle ID', y='Speed (km/h)', data=df)
plt.title('Speed Distribution per Vehicle')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("results/speed_per_vehicle.png")
plt.show()

# Plot 2: Speed Over Time
plt.figure(figsize=(10, 5))
df_sorted = df.sort_values(by='Timestamp')
sns.lineplot(x='Timestamp', y='Speed (km/h)', data=df_sorted)
plt.title('Speed Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/speed_over_time.png")
plt.show()

# Plot 3: Vehicle Count Over Time
plt.figure(figsize=(10, 5))
vehicle_counts = df.groupby('Timestamp')['Vehicle ID'].nunique()
vehicle_counts.plot(kind='bar')
plt.title('Vehicle Count per Timestamp')
plt.ylabel('Vehicle Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/vehicle_count.png")
plt.show()
