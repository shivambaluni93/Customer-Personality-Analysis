
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from PIL import Image

im = Image.open('icon.png')
st.set_page_config(page_title="Customer Cluster Prediction",page_icon=im)

st.image('main.png')

# Function to predict using the saved model
def predict_cluster(income, kidhome, teenhome, age, marital_status, education):
    model = pickle.load(open('gradient_boosting_1.pkl', 'rb'))
    features = [[income, kidhome, teenhome, age, marital_status, education]]
    prediction = model.predict(features)
    return prediction

# Create a Streamlit app
st.title('Customer Personality Analysis')
st.write('This app predicts the cluster of customers based on user input.')


# Add input widgets for user input
income = st.slider('Income', min_value=0, max_value=600000, value=50000, step=1000)
st.divider()
kidhome = st.radio ( "Select Number Of Kids In Household", ('0', '1','2') )
st.divider()
teenhome = st.radio ( "Select Number Of Teens In Household", ('0', '1','2') )
st.divider()
age = st.slider('Age', min_value=18, max_value=100, value=30, step=1)
st.divider()
marital_status = st.radio ( "Livig With Partner?", ('Yes', 'No') )
st.divider()
education = st.selectbox('Education', ['Basic', 'Graduation', 'Master', 'PhD'])

# Map marital status and education to numerical values
marital_status_map = {'Yes': 0, 'No': 1}
education_map = {'Basic': 0, 'Graduation': 1, 'Master': 2, 'PhD': 3}

marital_status_num = marital_status_map[marital_status]
education_num = education_map[education]
def main():
# Make a prediction and display the result
    if st.button('Predict'):
        cluster = predict_cluster(income, kidhome, teenhome, age, marital_status_num, education_num)
        st.write('Predicted Cluster:', f"The custumer belongs to cluster: {cluster}")

if __name__ == '__main__':
    main()

