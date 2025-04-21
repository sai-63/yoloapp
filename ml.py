import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv('expenses.csv')

# Strip any leading/trailing whitespace in the column names
df.columns = df.columns.str.strip()
df = df.rename(columns={'Expense': 'expense', 'Category': 'category', 'Date': 'date'})
print(df.columns)

# Check if the 'date' column exists and load it correctly
if 'date' not in df.columns:
    print("The 'date' column is missing or incorrectly named.")
    exit()

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Check for rows where 'date' could not be converted
if df['date'].isnull().any():
    print("Some dates could not be parsed. These rows will be dropped.")
    df = df.dropna(subset=['date'])

# Extract year and month from the 'date' column to create a new feature
df['year_month'] = df['date'].dt.to_period('M')

# Sort the dataframe by category and date to calculate past expenses
df = df.sort_values(by=['category', 'date'])

# Create lag features for the last 3 months of expenses for each category
df['expense_last_month'] = df.groupby('category')['expense'].shift(1)
df['expense_two_months_ago'] = df.groupby('category')['expense'].shift(2)
df['expense_three_months_ago'] = df.groupby('category')['expense'].shift(3)

# Drop rows where there are missing values for the lag features
df = df.dropna(subset=['expense_last_month', 'expense_two_months_ago', 'expense_three_months_ago'])

# Initialize a dictionary to store models for each category
category_models = {}

# Loop through each unique category to train a model
for category in df['category'].unique():
    category_data = df[df['category'] == category]
    
    # Skip categories with insufficient data (less than 4 data points)
    if len(category_data) < 4:
        print(f"Not enough data to train a model for category: {category}")
        continue
    
    # Define the feature set (past 3 months' expenses) and the target variable (current month's expense)
    X = category_data[['expense_last_month', 'expense_two_months_ago', 'expense_three_months_ago']]
    y = category_data['expense']
    
    # Split the data into training and test sets (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize and train the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the Mean Absolute Error (MAE) to evaluate the model's performance
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Category: {category}, MAE: {mae}")
    
    # Store the trained model in the dictionary
    category_models[category] = model

# Function to predict the next month's expense for a given category
def predict_next_month_expense(category, last_month_expense, two_months_ago_expense, three_months_ago_expense):
    # Check if a model exists for the given category
    if category not in category_models:
        print(f"No model found for category {category} (might have insufficient data).")
        return None
    
    # Get the model for the specified category
    model = category_models[category]
    
    # Prepare the feature set for prediction
    X_new = np.array([[last_month_expense, two_months_ago_expense, three_months_ago_expense]])
    
    # Make the prediction for the next month's expense
    next_month_expense = model.predict(X_new)
    return next_month_expense[0]

# Example usage: Predict next month's expense for a category (e.g., 'Grocery')
category = 'Grocery'
last_month_expense = 200  # Example expense for the last month
two_months_ago_expense = 150  # Example expense for two months ago
three_months_ago_expense = 180  # Example expense for three months ago

# Predict the next month's expense for the specified category
predicted_expense = predict_next_month_expense(category, last_month_expense, two_months_ago_expense, three_months_ago_expense)
if predicted_expense is not None:
    print(f"Predicted expense for {category} next month: ${predicted_expense:.2f}")
