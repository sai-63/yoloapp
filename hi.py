import csv
import random
from datetime import datetime, timedelta

# Categories for expenses
categories = ["Grocery", "Entertainment", "Subscription", "Transport", "Dining"]

# Generate random data
data = []
for i in range(100):
    expense = round(random.uniform(50, 5000), 2)  # Random expense between 50 and 5000
    category = random.choice(categories)  # Random category
    date = datetime.now() - timedelta(days=random.randint(0, 365))  # Random date within the past year
    data.append([expense, category, date.strftime("%Y-%m-%d")])

# Write to a CSV file
with open("expenses10.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Expense", "Category", "Date"])  # Write header
    writer.writerows(data)  # Write records

print("CSV file 'expenses10.csv' created successfully!")
