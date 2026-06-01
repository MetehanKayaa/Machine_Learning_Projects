import pandas as pd
import streamlit as st
import numpy as np
import requests
import pickle
import plotly.graph_objects as go
# style sheet
st.markdown("""
<style>
    /* Sağdaki ana kutu tasarımı */
    .prediction-container {
        background-color: #778da9; /* Görseldeki gri-mavi ton */
        padding: 25px;
        border-radius: 15px;
        color: white;
        font-family: 'sans serif';
    }
    
    /* Tahmin butonu (Benign/Malignant) */
    .diagnosis-label {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 10px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .benign-label { background-color: #28a745; color: white; }
    .malignant-label { background-color: #dc3545; color: white; }

    /* Sayıların olduğu siyah kutucuklar */
    .prob-value {
        background-color: #1b263b;
        color: #52b788; /* Matrix yeşili gibi bir ton */
        padding: 5px 10px;
        border-radius: 5px;
        font-family: monospace;
        display: block;
        margin-top: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)



#data
df=pd.read_pickle('./predict cancer/cleaned_df.pkl')





#sidebar
def add_sidebar():

    st.sidebar.header('Cell Nuclei Measurements')
    data=df
    
    #gptden aldım özellikler için sidebar eklentileri
    #label ve column label appde gözükücek columnda datadan çekicek
    slider_labels = [
        ("Radius (mean)", "radius_mean"), 
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]


    input_dict={}


    for label, key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict





#scaler yapma
def get_scaled_values(input_dict):
    data=df
    
    X=data.drop(columns=['diagnosis'])
    
    scaled_dict={}

    for key,value in input_dict.items():
        max_val=X[key].max() # girilen değerin sütünuna git maksını çek
        min_val=X[key].min()
        scaled_value=(value-min_val)/(max_val-min_val)
        scaled_dict[key]=scaled_value
    return scaled_dict







# radar chart (multiple trace radar chart from plotly documentation)
def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  # se mean ve worstü olan featureslar
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig
    


#predictions part

def add_predictions(input_data):
    model_path = 'predict cancer/logistic_model.pkl'
    scaler_path = 'predict cancer/standard_scaler.pkl'
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    probabilities = model.predict_proba(input_array_scaled)
    
    diag_class = "benign-label" if prediction[0] == 0 else "malignant-label"
    diag_text = "Benign" if prediction[0] == 0 else "Malignant"
    
    # HTML bloğunu en sola yasla (Hiç boşluk bırakma)
    st.markdown(f"""
<div class="prediction-container">
<h2 style="color: white; margin-bottom: 20px;">Cell cluster prediction</h2>
<p style="margin-bottom: 5px;">The cell cluster is:</p>
<div class="diagnosis-label {diag_class}">{diag_text}</div>
<p style="margin-top: 15px; margin-bottom: 5px;">Probability of being benign:</p>
<div class="prob-value">{probabilities[0][0]:.15f}</div>
<p style="margin-top: 15px; margin-bottom: 5px;">Probability of being malicious:</p>
<div class="prob-value">{probabilities[0][1]:.15f}</div>
<p style="font-size: 13px; margin-top: 25px; line-height: 1.4; color: #e0e1dd;">
This app can assist medical professionals in making a diagnosis, 
but should not be used as a substitute for a professional diagnosis.
</p>
</div>
""", unsafe_allow_html=True)
    
    
    
    
    
#üst icon
st.set_page_config(
    page_title='Breast Cancer Predictor',
    page_icon="👩‍⚕️",
    layout='wide',
    initial_sidebar_state='expanded'
)










input_data =add_sidebar() #sidebar en sonunda çzellikler return ediyor






with st.container():
    st.title("Breast Cancer Predictor")
    st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")


col1,col2 = st.columns([4,1])

with col1:
    radar_chart=get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
with col2:
      add_predictions(input_data)
    










































