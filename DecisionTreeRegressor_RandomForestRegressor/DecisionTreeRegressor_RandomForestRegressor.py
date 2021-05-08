# Import the necessary modules
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Read the file
FILE = 'train.csv'
df_home = pd.read_csv(FILE)

# Response variable
y = df_home.SalePrice

# Features 
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = df_home[features]

# Split and train dataset
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Specify model
dt_model = DecisionTreeRegressor(random_state = 1)

# Fit model
dt_model.fit(train_X, train_y)

# Validate predictions
preds = dt_model.predict(val_X)
val_mae = mean_absolute_error(preds, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
dt_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
dt_model.fit(train_X, train_y)
val_predictions = dt_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state = 1)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)

# file you will use for predictions
test_data_path = 'test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})
output.to_csv('preds.csv', index = False)