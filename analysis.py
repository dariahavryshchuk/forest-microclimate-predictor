import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
import joblib


df = pd.read_csv("data/Processed_microclimate_data.csv")

# print(df.head())

# print(df.shape)
# print(df.columns)
# print(df.dtypes)

#print(df.isnull().sum())

#print(df.describe())

#print(df.duplicated().sum())

#print(df["Canopy"].min())
# print(df["Canopy"].max())


#print(df["Date"].dtype)

#print(df["Location"].unique())
#print(df["Site"].unique())


df["TempBuffer"] = df["MetMaxTemp"] - df["MaxTemp"]
#print(df["TempBuffer"].describe())

#print(df.groupby("Canopy")["TempBuffer"].mean())
#print(df["Canopy"].corr(df["TempBuffer"]))
#print(df.groupby("SensorHeight")["TempBuffer"].mean())


plt.figure()
plt.scatter(df["Canopy"], df["TempBuffer"])
plt.xlabel("Canopy Cover (%)")
plt.ylabel("Temperature Buffering (C)")
plt.title("Canopy Cover vs Temperature Buffering")
# plt.show() 

canopy_avg = df.groupby("Canopy")["TempBuffer"].mean()

plt.figure
plt.plot(canopy_avg.index, canopy_avg.values, marker="o")
plt.xlabel("Canopy Cover (%)")
plt.ylabel("Average Temperature Buffering (C)")
plt.title("Average Temperature Buffering by Canopy Cover")
# plt.show()

height_avg = df.groupby("SensorHeight")["TempBuffer"].mean()
plt.figure()
plt.bar(height_avg.index, height_avg.values)
plt.xlabel("Sensor Height")
plt.ylabel("Average Temperature Buffering (C)")
plt.title("Temperature Buffering by Sensor Height")
# plt.show()


#print(df["MetMaxTemp"].corr(df["TempBuffer"]))
#print(df.groupby("Location")["TempBuffer"].mean())

df["Elevation"] = df["Site"].str[-2:]
#print(df["Elevation"].unique())
#print(df.groupby("Elevation")["TempBuffer"].mean())

#print(df.groupby("Elevation")["Canopy"].mean())

df["SensorHeightEncoded"] = df["SensorHeight"].map({"High": 0, "Low": 1})
df["ElevationEncoded"] = df["Elevation"].map({"Hi": 0, "Lo": 1})
X = df[["Canopy", "SensorHeightEncoded", "ElevationEncoded"]]
y = df["TempBuffer"]
#print(X.head())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#print(X_train.shape)
#print(X_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
#print(predictions[:5])

mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

baseline_value = y_train.mean()
print("Baseline prediction:", baseline_value)

baseline_mae = (y_test - baseline_value).abs().mean()
print("Baseline MAE:", baseline_mae)


rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)
print("Random Forest MAE:", rf_mae)

lr_rmse = root_mean_squared_error(y_test, predictions)
print("Linear Regression RMSE:", lr_rmse)

rf_rmse = root_mean_squared_error(y_test, rf_predictions)
print("Random Forest RMSE:", rf_rmse)

lr_r2 = r2_score(y_test, predictions)
print("Linear Regression R2:", lr_r2)

rf_r2 = r2_score(y_test, rf_predictions)
print("Random Forest R2:", rf_r2)




for feature, importance in zip(X.columns, rf_model.feature_importances_):
    print(feature, importance)

feature_names = ["Canopy Cover", "Sensor Height", "Elevation"]
plt.figure()
plt.bar(feature_names, rf_model.feature_importances_)
plt.title("Random Forest Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
# plt.show()


df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
print(df.groupby("Month")["TempBuffer"].mean())

X_improved = df[[
    "Canopy",
    "SensorHeightEncoded",
    "ElevationEncoded",
    "Month"
]]

X_train_improved, X_test_improved, y_train_improved, y_test_improved = train_test_split(
    X_improved,
    y,
    test_size=0.2,
    random_state=42
)

#print(X_train_improved.shape)
#print(X_test_improved.shape)

rf_improved = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_improved.fit(X_train_improved, y_train_improved)
joblib.dump(rf_improved, "forest_model.pkl")
rf_improved_predictions = rf_improved.predict(X_test_improved)
rf_improved_mae = mean_absolute_error(y_test_improved, rf_improved_predictions)
print("Improved Random Forest MAE:", rf_improved_mae)

df["DayOfYear"] = df["Date"].dt.dayofyear
print(df[["Date", "Month", "DayOfYear"]].head())

X_day = df[[
    "Canopy",
    "SensorHeightEncoded",
    "ElevationEncoded",
    "DayOfYear"
]]


X_train_day, X_test_day, y_train_day, y_test_day = train_test_split(
    X_day,
    y,
    test_size=0.2,
    random_state=42
)
rf_day = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_day.fit(X_train_day, y_train_day)
rf_day_predictions = rf_day.predict(X_test_day)
rf_day_mae = mean_absolute_error(y_test_day, rf_day_predictions)
print("DayOfYear Random Forest MAE:", rf_day_mae)

rf_improved_rmse = root_mean_squared_error(y_test_improved, rf_improved_predictions)
print("Improved Random Forest RMSE:", rf_improved_rmse)
rf_improved_r2 = r2_score(y_test_improved, rf_improved_predictions)
print("Improved Random Forest R2:", rf_improved_r2)



# location experiment
location_encoded = pd.get_dummies(df["Location"], prefix="Location", dtype=int)
print(location_encoded.head())

X_location = pd.concat([X_improved, location_encoded], axis=1)
print(X_location.head())
print(X_location.shape)

X_train_location, X_test_location, y_train_location, y_test_location = train_test_split(
    X_location,
    y,
    test_size=0.2,
    random_state=42
)

rf_location = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_location.fit(X_train_location, y_train_location)
rf_location_predictions = rf_location.predict(X_test_location)
rf_location_mae = mean_absolute_error(
    y_test_location,
    rf_location_predictions
)

print("Location Random Forest MAE:", rf_location_mae)


group_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
groups = df["Sensor"]
train_idx, test_idx = next(
    group_split.split(X_improved, y, groups=groups)
)
print(len(train_idx))
print(len(test_idx))

X_train_group = X_improved.iloc[train_idx]
X_test_group = X_improved.iloc[test_idx]
y_train_group = y.iloc[train_idx]
y_test_group = y.iloc[test_idx]
train_sensors = set(df["Sensor"].iloc[train_idx])
test_sensors = set(df["Sensor"].iloc[test_idx])

print("Training sensors:", len(train_sensors))
print("Testing sensors:", len(test_sensors))
print("Sensor overlap:", train_sensors.intersection(test_sensors))


rf_group = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_group.fit(X_train_group, y_train_group)
rf_group_predictions = rf_group.predict(X_test_group)
rf_group_mae = mean_absolute_error(y_test_group, rf_group_predictions)
print("Grouped Random Forest MAE:", rf_group_mae)

rf_group_rmse = root_mean_squared_error(
    y_test_group,
    rf_group_predictions
)

print("Grouped Random Forest RMSE:", rf_group_rmse)

rf_group_r2 = r2_score(
    y_test_group,
    rf_group_predictions
)
print("Grouped Random Forest R2:", rf_group_r2)


group_cv = GroupShuffleSplit(
    n_splits=5,
    test_size=0.2,
    random_state=42
)
group_maes = []
group_r2s = []

for train_idx, test_idx in group_cv.split(X_improved, y, groups=groups):
    X_train_cv = X_improved.iloc[train_idx]
    X_test_cv = X_improved.iloc[test_idx]
    y_train_cv = y.iloc[train_idx]
    y_test_cv = y.iloc[test_idx]

    rf_cv = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf_cv.fit(X_train_cv, y_train_cv)
    cv_predictions = rf_cv.predict(X_test_cv)

    group_maes.append(
        mean_absolute_error(y_test_cv, cv_predictions)
    )

    group_r2s.append(
        r2_score(y_test_cv, cv_predictions)
    )
print("Grouped MAEs:", group_maes)
print("grouped R2s:", group_r2s)
print("Average Grouped MAE:", sum(group_maes) / len(group_maes))
print("Average Grouped R2:", sum(group_r2s) / len(group_r2s))



plt.figure()
plt.scatter(
    y_test_improved,
    rf_improved_predictions,
    alpha=0.4
)

plt.xlabel("Actual Temperature Buffering (Celsius)")
plt.ylabel("Predicted Temperature Buffering (Celsius)")
plt.title("Actual vs Predicted Temperature Buffering")
min_value = min(y_test_improved.min(), rf_improved_predictions.min())
max_value = max(y_test_improved.max(), rf_improved_predictions.max())
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)
plt.legend()
plt.show()