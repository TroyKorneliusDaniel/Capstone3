import streamlit as st
import pandas as pd
import pickle

# Judul Utama
st.title('Survival Rate Predictor')
st.text('This web can be used to predict your survival rate')

# Menambahkan sidebar
st.sidebar.header("Please input your features")

def create_user_input():
    # Categorical Features
    job = st.sidebar.selectbox('Job', ['admin', 'self-employed', 'services', 'housemaid', 'technician', 'management', 'student', 'blue-collar', 'entrepreneur', 'retired', 'unemployed'])
    housing = st.sidebar.radio('Housing', ['no', 'yes'])
    loan = st.sidebar.radio('Loan', ['no', 'yes'])
    contact = st.sidebar.radio('Contact', ['cellular', 'telephone'])
    month = st.sidebar.selectbox('Month', ['jun', 'apr', 'may', 'nov', 'jan', 'sep', 'feb', 'mar', 'aug', 'jul', 'oct', 'dec'])
    poutcome = st.sidebar.selectbox('Poutcome', ['unknown', 'other', 'failure', 'success'])
    
    # Numerical Features
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
    campaign = st.sidebar.number_input('Campaign', min_value=1, max_value=50, value=1)
    pdays = st.sidebar.number_input('Pdays', min_value=-1, max_value=1000, value=-1)

    # Creating a dictionary with user input
    user_data = {
        'age': age,
        'job': job,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'poutcome': poutcome,
        'campaign': campaign,
        'pdays': pdays,
    }
    
    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get customer data
data_customer = create_user_input()

# Membuat 2 kontainer
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

# Load model
with open('Model Final.sav', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict to data
kelas = model_loaded.predict(data_customer)
probability = model_loaded.predict_proba(data_customer)[0]  # Get the probabilities

# Menampilkan hasil prediksi
with col2:
    st.subheader('Prediction Result')
    if kelas == 1:
        st.write('Class 1: This customer will Deposit')
    else:
        st.write('Class 2: This customer will not Deposit')
    
    # Displaying the probability of the customer surviving
    st.write(f"Probability of Deposit: {probability[1]:.2f}")
