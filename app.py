import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
df = pd.read_csv("chocolate_sales.csv")
print(df.head())
# Convert Date to datetime and extract useful parts
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Drop original Date
df = df.drop(columns=["Date"])

X = df.drop("Amount", axis=1)   # input features
y = df["Amount"]                # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cat_cols = ["Country", "Product", "Sales Person"]
num_cols = ["Boxes Shipped", "Year", "Month"]

# Preprocess
cat_encoder = OneHotEncoder(handle_unknown="ignore")
transformer = ColumnTransformer([
    ("cat", cat_encoder, cat_cols)
], remainder="passthrough")

# Model
model = RandomForestRegressor(random_state=42)

# Entire pipeline
clf = Pipeline([
    ("prep", transformer),
    ("model", model)
])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
