import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset with Latin-1 encoding
def load_data():
    data = pd.read_csv('rewards.csv')
    return data

# Function to train the Random Forest Classifier
def train_random_forest(data):
    X = data.drop('Reward', axis=1)
    y = data['Reward']

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_train

# Function to predict rewards for user input
def predict_reward(input_data, model, X_train):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    
    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[X_train.columns]
    predicted_reward = model.predict(input_df)[0]
    return predicted_reward

# Main function
def main():
    st.title("Reward Prediction Application")
    data = load_data()
    
    # Train the Random Forest Classifier
    model, X_train = train_random_forest(data)

    # Display input fields
    st.header("User Input")
    frequency = st.number_input("Frequency", min_value=1, step=1)
    total_spending = st.number_input("Total Spending", min_value=0, step=100)
    product_type = st.text_input("Most Frequent Product Type", value="")
    brand = st.text_input("Most Frequent Brand", value="")

    # Submit button
    if st.button("Submit"):
        user_input = {
            'Frequency': frequency,
            'Total_Spending': total_spending,
            'Most_Frequent_Product_Type': product_type,
            'Most_Frequent_Brand': brand
        }

        # Predict reward for the user input
        predicted_reward = predict_reward(user_input, model, X_train)
        st.write(f"Predicted Reward for User Input: {predicted_reward}")

if __name__ == "__main__":
    main()
