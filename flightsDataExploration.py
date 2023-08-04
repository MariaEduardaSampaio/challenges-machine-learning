import pandas as pd
from matplotlib import pyplot as plt


def show_distribution(var_data):
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    fig, ax = plt.subplots(2, 1, figsize=(10, 4))

    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    ax[0].axvline(x=min_val, color='green', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=mean_val, color='yellow', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=med_val, color='cyan', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=mod_val, color='gray', linestyle='dashed', linewidth=2)
    ax[0].axvline(x=max_val, color='red', linestyle='dashed', linewidth=2)

    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    fig.suptitle(var_data.name)
    plt.show()


df_flights = pd.read_csv('data/flights.csv')
# print(df_flights.head())

# Cleaning the data
print(df_flights.isnull().sum())
df_flights.DepDel15 = df_flights.DepDel15.fillna(0)
print(df_flights.isnull().sum())

# Eliminating outliers
fields = ['DepDelay', 'ArrDelay']
for col in fields:
    show_distribution(df_flights[col])

# Maintain values between 1% and 90% of original data
arrDelay_01pcntile = df_flights.ArrDelay.quantile(0.01)
arrDelay_90pcntile = df_flights.ArrDelay.quantile(0.90)
df_flights = df_flights[df_flights.ArrDelay < arrDelay_90pcntile]
df_flights = df_flights[df_flights.ArrDelay > arrDelay_01pcntile]

depDelay_01pcntile = df_flights.DepDelay.quantile(0.01)
depDelay_90pcntile = df_flights.DepDelay.quantile(0.90)
df_flights = df_flights[df_flights.DepDelay < depDelay_90pcntile]
df_flights = df_flights[df_flights.DepDelay > depDelay_01pcntile]

# Exploring the data

for col in fields:
    show_distribution(df_flights[col])
    df_flights.boxplot(column=col, by='Carrier',  figsize=(8, 8))
    plt.show()

print(df_flights[fields].describe())

# Which departure airport has the highest average departure delay?
departure_airport_group = df_flights.groupby(df_flights.OriginAirportName)

mean_departure_delays = pd.DataFrame(departure_airport_group.DepDelay.mean())
mean_departure_delays = mean_departure_delays.sort_values('DepDelay', ascending=False)
mean_departure_delays.plot(kind='bar', figsize=(10, 12))
plt.show()


