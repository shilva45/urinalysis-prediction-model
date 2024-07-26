import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def encode(data):
    if data["Gender"] == "Female":
        data["Gender"] = 0
    else:
        data["Gender"] = 1

    color_map = {
        "AMBER": 0, "DARK YELLOW": 2, "LIGHT YELLOW": 4, 
        "STRAW": 8, "YELLOW": 9
    }
    transparency_map = {
        "CLEAR": 0, "CLOUDY": 1, "HAZY": 2, 
        "SLIGHTLY HAZY": 3, "TURBID": 4
    }
    glucose_map = {
        "1+": 0, "2+": 1, "3+": 2, "4+": 3, 
        "NEGATIVE": 4, "TRACE": 5
    }
    protein_map = {
        "1+": 0, "2+": 1, "3+": 2, "NEGATIVE": 3, "TRACE": 4
    }
    wbc_map = {
        "0-1": 0, "0-2": 1, "1-2": 4, "1-3": 5, "2-4": 28, "6-8": 59
    }
    rbc_map = {
        "0-1": 0, "0-2": 1, "1-2": 4, "1-3": 5, "2-4": 28, "6-8": 59
    }
    epithelial_map = {
        "FEW": 0, "MODERATE": 2, "NONE SEEN": 3, 
        "OCCASIONAL": 4, "PLENTY": 5, "RARE": 6
    }
    mucous_map = {
        "FEW": 0, "MODERATE": 1, "NONE SEEN": 2, 
        "OCCASIONAL": 3, "PLENTY": 4, "RARE": 5
    }
    amorphous_map = {
        "FEW": 0, "MODERATE": 1, "NONE SEEN": 2, 
        "OCCASIONAL": 3, "PLENTY": 4, "RARE": 5
    }
    bacteria_map = {
        "FEW": 0, "MODERATE": 2, "OCCASIONAL": 3, 
        "PLENTY": 4, "RARE": 5
    }

    data["Color"] = color_map.get(data["Color"], 0)
    data["Transparency"] = transparency_map.get(data["Transparency"], 0)
    data["Glucose"] = glucose_map.get(data["Glucose"], 4)
    data["Protein"] = protein_map.get(data["Protein"], 3)
    data["WBC"] = wbc_map.get(data["WBC"], 1)
    data["RBC"] = rbc_map.get(data["RBC"], 1)
    data["Epithelial Cells"] = epithelial_map.get(data["Epithelial Cells"], 6)
    data["Mucous Threads"] = mucous_map.get(data["Mucous Threads"], 2)
    data["Amorphous Urates"] = amorphous_map.get(data["Amorphous Urates"], 2)
    data["Bacteria"] = bacteria_map.get(data["Bacteria"], 5)
    
    return data

# Set up the Streamlit page
st.set_page_config(page_title='Urinary Tract Infection', page_icon=':medical_symbol:')
st.title(":medical_symbol: Urinary Tract Infection (UTI):medical_symbol:")
st.markdown("<h1 style='text-align: center;'>Prediction Model</h1>", unsafe_allow_html=True)

# Add an image
st.image("urineTest.jpg", caption="Urine Test", use_column_width=True)

st.write("""
Urinary Tract Infection (UTI) is a prevalent infection affecting the urinary system, primarily the bladder (cystitis) or potentially spreading to the kidneys (pyelonephritis). Recognizing and diagnosing UTI is crucial due to several reasons. First and foremost, early detection enables prompt treatment, which can prevent the infection from spreading and causing more severe complications such as kidney damage.
""")

st.write("""
Furthermore, accurate diagnosis allows healthcare providers to prescribe appropriate antibiotics tailored to the specific bacteria causing the infection, thereby improving treatment efficacy and reducing the risk of antibiotic resistance. Moreover, untreated or recurrent UTIs can lead to chronic issues and recurrent infections, impacting a person's quality of life and potentially leading to more serious health consequences. Therefore, understanding and diagnosing UTI promptly not only alleviates symptoms but also mitigates potential complications, ensuring better overall health outcomes for patients.""")

st.write("""
-------------------------------------------------------------------------------------------------------------------------------------------
""")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
color = st.selectbox("Color", ["YELLOW", "LIGHT YELLOW", "DARK YELLOW", "STRAW", "AMBER"])
transparency = st.selectbox("Transparency", ["CLEAR", "SLIGHTLY HAZY", "HAZY", "TURBID", "CLOUDY"])
glucose = st.selectbox("Glucose", ["NEGATIVE", "1+", "2+", "3+", "TRACE"])
protein = st.selectbox("Protein", ["NEGATIVE", "1+", "2+", "3+", "TRACE"])
wbc = st.selectbox("WBC", ["0-1", "0-2", "1-2", "1-3", "2-4", "6-8"])
rbc = st.selectbox("RBC", ["0-1", "0-2", "1-2", "1-3", "2-4", "6-8"])
epithelial_cells = st.selectbox("Epithelial Cells", ["RARE", "FEW", "MODERATE", "PLENTY", "OCCASIONAL", "NONE SEEN"])
mucous_threads = st.selectbox("Mucous Threads", ["NONE SEEN", "FEW", "RARE", "MODERATE", "PLENTY"])
amorphous_urates = st.selectbox("Amorphous Urates", ["NONE SEEN", "FEW", "RARE", "MODERATE", "PLENTY"])
bacteria = st.selectbox("Bacteria", ["RARE", "FEW", "MODERATE", "PLENTY", "OCCASIONAL"])

# Data dictionary
data = {
    "Age": age,
    "Gender": gender,
    "Color": color,
    "Transparency": transparency,
    "Glucose": glucose,
    "Protein": protein,
    "WBC": wbc,
    "RBC": rbc,
    "Epithelial Cells": epithelial_cells,
    "Mucous Threads": mucous_threads,
    "Amorphous Urates": amorphous_urates,
    "Bacteria": bacteria
}

st.write("""
-------------------------------------------------------------------------------------------------------------------------------------------
""")

st.subheader("Your Input Data")
with st.expander("Click to see your input data"):
    st.write(data)

# Encode and prepare the data
encoded_data = encode(data)
df = pd.DataFrame([encoded_data])

# Prediction button
if st.button("Predict"):
    try:
        with open("random_forest_model.pkl", "rb") as file:
            model = pickle.load(file)
            
        prediction = model.predict(df)
        print(prediction)
        result = "Result is Positive" if prediction[0] == 1 else "Result is Negative"
        st.success(result)
    except FileNotFoundError:
        st.error("Model file not found.")
    except pickle.UnpicklingError:
        st.error("Error unpickling the model file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
