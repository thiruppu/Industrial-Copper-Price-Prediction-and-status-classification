import streamlit as st
import streamlit_lottie
from streamlit_lottie import st_lottie_spinner
import streamlit_option_menu as option_menu
import base64
import json
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Function to encode an image to base64
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

st.set_page_config(layout="wide")

# Load images and convert them to base64
img_base64 = get_img_as_base64("fact.jpg")
img_base64_1 = get_img_as_base64("pred.png")
lottie_streamlit = load_lottiefile("business3.json")

def classification(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14):
    with open('knn_model.pkl', 'rb') as file:
        guvi = pickle.load(file)
    
    input_features = [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14]
    input_tuple = np.array(input_features)
    input_features_reshaped = input_tuple.reshape(1, -1)
    classi = guvi.predict(input_features_reshaped)
    return classi[0]

def regression(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14):

    with open('rf_model.pkl', 'rb') as file:
        guvi1 = pickle.load(file)
    input_features = [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14]
    input_tuple = np.array(input_features)
    #print(input_tuple)
    
    input_features_reshaped = np.array(input_tuple).reshape(-1,1)
    #print(input_features_reshaped.shape)
    input_features_reshaped = input_features_reshaped.T
    prediction = guvi1.predict(input_features_reshaped)
    return prediction[0] 


st.markdown(
    f"""
    <style>
    .css-1v3fvcr {{
        color: black;  /* Text color for the main content */
    }}
    /* Change the color of the selected menu item */
    .nav-link.active {{
        background-color: #784549 !important;  /* Selected option background color */
        color: white !important;  /* Selected option text color */
    }}
    /* Change the color of unselected menu items */
    .nav-link {{
        color: white !important;  /* Unselected option text color */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

col_1, col_2, col_3 = st.columns([1, 10, 1])

with col_2:
    st.title("Copper Industry Pricing & Lead Classification Tool")
    st.write(" ")
    menu = option_menu.option_menu(None, ["Overview", "Status Classification", "Price Prediction", "Contact Us"], 
                                   icons=['house', 'cart-check', 'currency-rupee'], 
                                   menu_icon="cast", default_index=0, orientation="horizontal")
    st.write(" ")

col1_1, col1_2, col1_3 = st.columns([1, 18, 1])
with col1_2:
    if menu == "Overview":
        col1, col2 = st.columns([5, 3])
        with col1:
            st.subheader("Status Classification Model:")
            st.subheader("Model Employed: :blue[K-Nearest Neighbors (KNN) Classification]")
            st.subheader("Accuracy Achieved: :green[99%] ")
            st.subheader("Tools Used : Python | Pandas | Machine Learning ")
            st.subheader("Objective:")
            st.write("The Lead Classification Model aims to revolutionize lead management in the copper industry by accurately predicting the likelihood of lead conversion. By classifying leads into categories of WON (likely to convert) or LOST (unlikely to convert), the model enhances sales efficiency and focuses resources on high-potential opportunities. It leverages historical data and advanced machine learning techniques to identify patterns and key indicators of successful conversions, thereby improving overall conversion rates. The model adapts to market fluctuations, provides data-driven insights for informed decision-making, and minimizes human bias, ensuring a consistent and objective evaluation of each lead. Additionally, it helps optimize sales strategies by highlighting characteristics of successful leads and facilitates continuous improvement through feedback analysis, driving more effective and targeted sales efforts.")
            st.write(" ")

        with col2: 
            st.title(" ")
            st.lottie(lottie_streamlit)

        st.divider()
        col3, col4 = st.columns([3, 5])
        with col3:
            st.subheader("Price Prediction Model:")
            st.lottie("https://lottie.host/457b6ec0-ce81-40b7-a2d1-a11ae2779254/wHrpImuyLd.json")
        with col4:
            st.subheader(" ")
            st.subheader("Model Employed: :blue[Random Forest Regression]")
            st.subheader("Accuracy Achieved: :green[96%] ")
            st.subheader("Tools Used : Python | Pandas | Machine Learning ")
            st.subheader("Objective:")
            st.write("The Price Prediction model is designed to accurately forecast the optimal selling price of copper products by analyzing a wide range of market and historical data. This advanced regression model integrates multiple critical factors that influence pricing, including real-time market demand, fluctuating supply conditions, raw material costs, and historical sales patterns.")
            st.write("By leveraging this model, the copper industry gains a powerful tool that not only predicts prices with high accuracy but also offers insights into the underlying market dynamics. This allows businesses to set competitive prices that are aligned with current market trends, maximizing profitability while mitigating risks associated with volatile market conditions. Furthermore, the model’s predictive capabilities support strategic decision-making by enabling companies to anticipate market shifts and adjust their pricing strategies proactively.")
            st.write(" ")

        st.divider()
        col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
        with col1_1:
            st.image("linkedin.png")
            st.write("[LinkedIn](https://www.linkedin.com/in/thiruppugazhan-s-277705282/)")
        with col1_2:
            st.image("instagram.png")
            st.write("[Instagram](https://instagram.com/_thiruppugazhan)")
        with col1_3:
            st.image("github.png")
            st.write("[GitHub](https://github.com/thiruppu)")

    elif menu == "Status Classification":
        st.markdown(
            f"""
            <style>
            .stApp {{
            background-color: #784549;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        col2_1, col2_2, col2_3, col2_4 = st.columns([1, 4, 4, 1])
        with col2_2:
            var1 = st.text_input("Quantity Tons")
            var2 = st.text_input("Customer ID")
            var3 = st.text_input("Country")
            var4 = st.text_input("Application")
            var5 = st.text_input("Thickness")
            var6 = st.text_input("Width")
            var7 = st.text_input("Product ref")
        
        with col2_3:
            var8 = st.text_input("Selling Price")
            var9 = st.text_input("Item Year")
            var10 = st.text_input("Item Month")
            var11 = st.text_input("Item Day")
            var12 = st.text_input("Delivery Year")
            var13 = st.text_input("Delivery Month")
            var14 = st.text_input("Delivery Day")
            
            button_classification = st.button("Classify")
        
        if button_classification:
            # Validate input and handle empty fields
            errors = []
            try:
                var1 = float(var1) if var1 else errors.append("Please enter Quantity in Tons")
                var2 = int(var2) if var2 else errors.append("Please enter Customer ID")
                var3 = int(var3) if var3 else errors.append("Please enter Country")
                var4 = int(var4) if var4 else errors.append("Please enter Application")
                var5 = float(var5) if var5 else errors.append("Please enter Thickness")
                var6 = float(var6) if var6 else errors.append("Please enter Width")
                var7 = int(var7) if var7 else errors.append("Please enter Product ref")
                var8 = float(var8) if var8 else errors.append("Please enter Selling Price")
                var9 = int(var9) if var9 else errors.append("Please enter Item Year")
                var10 = int(var10) if var10 else errors.append("Please enter Item Month")
                var11 = int(var11) if var11 else errors.append("Please enter Item Day")
                var12 = int(var12) if var12 else errors.append("Please enter Delivery Year")
                var13 = int(var13) if var13 else errors.append("Please enter Delivery Month")
                var14 = int(var14) if var14 else errors.append("Please enter Delivery Day")
                
                if not errors:
                    classi = classification(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14)
                    col3_1,col3_2,col3_3 = st.columns(3)
                    with col3_2:
                        if classi == 0:
                            st.subheader(f"Classification Result: {classi} == LOST")
                        else:
                            st.subheader(f"Classification Result: {classi} == WON")
                else:
                    for error in errors:
                        st.warning(error)
                    
            except ValueError as e:
                st.error(f"Invalid input: {e}")

    elif menu == "Price Prediction":
        st.markdown(
            f"""
            <style>
            .stApp {{
            background-color: #784549;  
            }}
            
            </style>
            """,
            unsafe_allow_html=True
        )
        col2_1, col2_2, col2_3, col2_4 = st.columns([1, 4, 4, 1])
        with col2_2:
            var1 = st.text_input("Quantity Tons")
            var2 = st.text_input("Customer ID")
            var3 = st.text_input("Country")
            var4 = st.text_input("Application")
            var5 = st.text_input("Thickness")
            var6 = st.text_input("Width")
            var7 = st.text_input("Product ref")
        
        with col2_3:
            var8 = st.text_input("Item Year")
            var9 = st.text_input("Item Month")
            var10 = st.text_input("Item Day")
            var11 = st.text_input("Delivery Year")
            var12 = st.text_input("Delivery Month")
            var13 = st.text_input("Delivery Day")
            var14 = st.text_input("Status")
            
            button_prediction = st.button("Predict")
        
        if button_prediction:
            # Validate input and handle empty fields
            errors = []
            try:
                var1 = float(var1) if var1 else errors.append("Please enter Quantity in Tons")
                var2 = int(var2) if var2 else errors.append("Please enter Customer ID")
                var3 = int(var3) if var3 else errors.append("Please enter Country")
                var4 = int(var4) if var4 else errors.append("Please enter Application")
                var5 = float(var5) if var5 else errors.append("Please enter Thickness")
                var6 = float(var6) if var6 else errors.append("Please enter Width")
                var7 = int(var7) if var7 else errors.append("Please enter Product ref")
                var8 = int(var8) if var8 else errors.append("Please enter Item Year")
                var9 = int(var9) if var9 else errors.append("Please enter Item Month")
                var10 = int(var10) if var10 else errors.append("Please enter Item Day")
                var11 = int(var11) if var11 else errors.append("Please enter Delivery Year")
                var12 = int(var12) if var12 else errors.append("Please enter Delivery Month")
                var13 = int(var13) if var13 else errors.append("Please enter Delivery Day")
                var14 = int(var14) if var14 else errors.append("Please enter Status")
                
                if not errors:
                    col4_1,col4_2,col4_3 = st.columns(3)
                    with col4_2:
                        predict = regression(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14)
                        
                        st.subheader(f"Predictive Selling Price is : {predict}")
                        
                else:
                    for error in errors:
                        st.warning(error)
                    
            except ValueError as e:
                st.error(f"Invalid input: {e}")
    elif menu == "Contact Us":
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #784549;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        col5_1,col5_2,col5_3 = st.columns(3)
        with col5_2:
            st.text_input("Name")
            st.text_input("E-Mail")
            st.text_input("Mobile")
            st.text_area("Message")
            st.button("Submit")