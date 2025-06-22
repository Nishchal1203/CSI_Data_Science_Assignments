#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# I am loading the dataset here we have the tw diff datasets for train and test the model 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print(train.head())
print(train.info())
print(train.describe())

# Now lets do the data preproecessing

# Drop columns with too many missing values
missing_thresh = 0.4 * len(train)
train = train.dropna(thresh=missing_thresh, axis=1)
test = test[train.columns.drop(['SalePrice'])]  


numeric_cols = train.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    train[col] = train[col].fillna(train[col].mean())
    if col in test.columns:
        test[col] = test[col].fillna(train[col].mean())


cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    if col in test.columns:
        test[col] = test[col].fillna(train[col].mode()[0])


full_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)
full_data = pd.get_dummies(full_data)


X_train = full_data.iloc[:train.shape[0], :]
X_test = full_data.iloc[train.shape[0]:, :]
y_train = train['SalePrice']


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_tr, y_tr)


y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE:", rmse)

test_preds = model.predict(X_test)


submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": test_preds
})
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv file saved successfully.")
