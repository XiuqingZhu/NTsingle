import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import os
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import matplotlib.pyplot as plt
import time
from PIL import Image

@st.cache_data
def load_image(image_path):
    return Image.open(image_path)

# Load and display the logo
logo = load_image("./logo.png")
st.image("./logo.png")

def get_fingerprints(smiles):
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        # Calculate MACCS fingerprints
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_bits = np.array(maccs_fp, dtype=int).tolist()

        # Calculate ECFP4 fingerprints
        ecfp4_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        ecfp4_bits = np.array(ecfp4_fp, dtype=int).tolist()

        return maccs_bits, ecfp4_bits
    except Exception as e:
        st.write("**Invalid SMILES string. Unable to perform subsequent calculations. **")
        return None, None

def generate_feature_vector(smiles, feature_order):
    maccs_bits, ecfp4_bits = get_fingerprints(smiles)
    if maccs_bits is None or ecfp4_bits is None:
        return None

    feature_vector = []
    for feature in feature_order:
        if feature.startswith("MACCS_"):
            index = int(feature.split("_")[1]) 
            feature_vector.append(maccs_bits[index])
        elif feature.startswith("ECFP4_bitvector"):
            index = int(feature.split("bitvector")[1])
            feature_vector.append(ecfp4_bits[index])

    return feature_vector

def run_progress():
    progress_bar = st.empty()
    for i in range(10):
        progress_bar.progress(i / 10, 'Progress')
        time.sleep(0.5)
    with st.spinner('Loading...'):
        time.sleep(2)
    progress_bar.empty()

st.write("Supported by the service of Zhu's AI-Drug Lab at the affiliated Brain Hospital, Guangzhou Medical University. If you have any questions, please contact me at 2018760376@gzhmu.edu.cn.")

# Define feature names
feature_df = pd.read_csv('./features_for_ML.csv')
feature_names = feature_df['Features'].values.tolist()

# Load the model
model = joblib.load('./Model_final.joblib')

# Streamlit user interface
st.title("Nephrotoxic Component Predictor")

st.write("**Please enter a SMILE string for predicting nephrotoxic components.**")

# Smiles: string input
smiles = st.text_input("SMILE:", value="")

if st.button("Predict"):
    # Generate feature vector
    feature_vector = generate_feature_vector(smiles, feature_names)
    
    if feature_vector is None:
        st.write("**Please provide a correct SMILES notation. **")
    else:
        features = np.array([feature_vector])
        
        # Predict class and probabilities
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display a separator line
        st.write("---") 
        # Display prediction results
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Prediction Probabilities:** {predicted_proba}")

        # Generate advice based on prediction results
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 1:
            advice = (
                f"According to our model, the compound you submitted poses a high risk of nephrotoxicity. "
                f"The model predicts that your likelihood of experiencing nephrotoxicity is {probability:.2f}%. "
                "While this is only an estimation, it indicates that the compound may be at a significant risk. "
            )
        else:
            advice = (
                f"According to our model, the compound you submitted has a low risk of nephrotoxicity. "
                f"The model predicts that your likelihood of not experiencing nephrotoxicity is {probability:.2f}%. "
            )

        st.write(advice)

        # Display a separator line
        st.write("---") 
        # Display SHAP values for each feature
        st.write("**SHAP values for each feature:**")
        
        run_progress()

        # Calculate SHAP values and display force plot    
        explainer = shap.TreeExplainer(model)  
        shap_values = explainer.shap_values(pd.DataFrame([feature_vector], columns=feature_names))

        shap_df = pd.DataFrame(shap_values[0].T, columns=["SHAP value"], index=feature_names)
        shap_df["Absolute SHAP value"] = shap_df["SHAP value"].abs()
        st.write(shap_df)
       
        st.write("---") 
        st.write("**The generated SHAP force plot of this compound:**")
        
        run_progress()
        
        # Delete old SHAP force plot image file (if it exists)
        if os.path.exists("./shap_force_plot.png"):
            os.remove("./shap_force_plot.png")

        # Choose the index of the output you want to visualize
        output_index = 0  # Adjust this index based on your specific needs

        # Force plot for the specified output
        expected_value = explainer.expected_value[output_index]

        positive_color = "#FF8C69"
        negative_color = "#B23AEE"
            
        # Generate the force plot for the specified output
        shap.force_plot(
            expected_value,
            shap_values[output_index],
            feature_names=pd.DataFrame([feature_vector], columns=feature_names).columns,
            matplotlib=True,
            show=True,
            plot_cmap=[positive_color, negative_color]
        )

        # Save and display the image
        plt.savefig("./shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("./shap_force_plot.png")

        run_progress()
        
        # Display features of this compound
        st.write("---")
        st.write("**The molecular fingerprints of this compound used in modeling:**")
        important_features = [feature_names[i] for i, value in enumerate(feature_vector) if value == 1]
        st.write(important_features)
