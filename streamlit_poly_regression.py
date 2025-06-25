import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.title("Polynomial Regression App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data")
    st.write(df)

    # Check if proper columns exist
    if df.shape[1] < 3:
        st.error("CSV should have at least 3 columns. Assuming column 2 is Position Level and column 3 is Salary.")
    else:
        x = df.iloc[:, 1:2].values #split the data as np array
        y = df.iloc[:, 2].values

        # Degree selector
        degree = st.slider("Select the Degree of the Polynomial", min_value=1, max_value=10, value=2) # we set the default polynomial degree as 2

        # Train model
        poly_reg = PolynomialFeatures(degree=degree)
        x_poly = poly_reg.fit_transform(x)

        model = LinearRegression()
        model.fit(x_poly, y)

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.scatter(x, y, color='red', label='Original Data')
        plt.plot(x, model.predict(x_poly), color='blue', label=f'Polynomial Regression (Degree {degree})')
        plt.title(f"Polynomial Regression (Degree {degree})")
        plt.xlabel("Position Level")
        plt.ylabel("Salary")
        plt.legend()
        st.pyplot(plt)

        # Input for prediction
        level = st.number_input("Enter the position level (float allowed):", min_value=0.0, max_value=10.0)

        if st.button("Predict Salary"):
            pred_salary = model.predict(poly_reg.transform([[level]]))[0]
            st.success(f"Predicted salary for level {level} is: ₹{pred_salary}")
