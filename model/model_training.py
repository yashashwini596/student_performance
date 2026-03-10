import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("../dataset/student_data.csv")

# Features
X = data[['study_hours','attendance','assignments','previous_marks']]

# Target
y = data['percentage']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train,y_train)

# Save model
joblib.dump(model,"student_model.pkl")

print("Model trained successfully!")