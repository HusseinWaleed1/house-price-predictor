import streamlit as st

# Page config
st.set_page_config(page_title="ℹ️ About - California House Price App", page_icon="ℹ️", layout="centered")

# Title
st.title("ℹ️ About This Project")

# Intro
st.markdown("""
## 🏠 California House Price Prediction App

This application predicts house prices in California using the **California Housing Dataset** from [Scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

### 📌 How it Works:
1. **Data Collection:**  
   We used the California Housing dataset which contains median house values for districts in California.
   
2. **Data Preprocessing:**  
   Features were standardized using `StandardScaler` to ensure better model performance.
   
3. **Model Training:**  
   A `RandomForestRegressor` model was trained to predict median house prices.

4. **Deployment:**  
   The trained model and scaler were saved using `joblib` and integrated with **Streamlit** for the web interface.

---

### 👨‍💻 Author:
- **Your Name Here**
- Data Science & Machine Learning Enthusiast
- GitHub: [Your GitHub Link](https://github.com/)
- LinkedIn: [Your LinkedIn Link](https://linkedin.com/)

---

### ⚙️ Tech Stack:
- **Python** 🐍  
- **Scikit-learn** 🤖  
- **Pandas & NumPy** 📊  
- **Streamlit** 🌐  
""")
