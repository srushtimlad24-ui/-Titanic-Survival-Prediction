# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

st.title("Titanic Survival Prediction - Logistic Regression")

# -----------------------------
# Upload Dataset
# -----------------------------

uploaded_file = st.file_uploader("Upload Titanic CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Dataset")
    st.dataframe(df.head())

    # -----------------------------
    # Data Preprocessing
    # -----------------------------

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df = df.dropna(subset=['Embarked'])

    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'Sex_male', 'Embarked_Q', 'Embarked_S']]
    y = df['Survived']

    # -----------------------------
    # Train-Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Model Training
    # -----------------------------

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"Accuracy: {round(accuracy, 3)}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # -----------------------------
    # Confusion Matrix Plot
    # -----------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    fig.colorbar(cax)

    for (i, j), val in pd.DataFrame(cm).stack().items():
        ax.text(j, i, int(val), ha='center', va='center')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    st.pyplot(fig)

else:
    st.info("Please upload the Titanic dataset CSV file.")
