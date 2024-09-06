from prophet import Prophet
import pandas as pd
import plotly.express as px

# Import data
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
test_data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")

# Format data correctly for Prophet
data_p = data[['Timestamp', 'trips']]
data_p.columns = ['ds', 'y']

data_p.head()

# Create and fit the model
model = Prophet()
modelFit = model.fit(data_p)

# Create test independent variable data
jan_2019 = test_data[["Timestamp"]]
jan_2019.columns = ['ds']
jan_2019.head()

# Create the forecast and store the prediction
forecast = model.predict(jan_2019)
forecast.head()
pred = forecast[['yhat']]
pred.head()
