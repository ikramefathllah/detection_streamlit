#import subprocess
#subprocess.call(['pip', 'install', '-r', 'https://github.com//ikramefathllah/detection_streamlit/requirements.txt'])
from unittest import result
import streamlit as st
import torch
from PIL import Image
from pathlib import Path
# Chemin vers le modèle YOLOv5
model_path = Path('model_paneau.pt').resolve()

# Charger le modèle
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))

names=['danger', 'no_entry', 'no_parking', 'no_stop', 'no_u_turn', 'parking', 'pedestrian', 'round', 'speed_limit_100_ar', 'speed_limit_100_en', 'speed_limit_120_ar', 'speed_limit_120_en', 'speed_limit_15_en', 'speed_limit_20_en', 'speed_limit_30_ar', 'speed_limit_30_en', 'speed_limit_40_en', 'speed_limit_50_ar', 'speed_limit_50_en', 'speed_limit_5_en', 'speed_limit_60_ar', 'speed_limit_60_en', 'speed_limit_70_ar', 'speed_limit_70_en', 'speed_limit_80_ar', 'speed_limit_80_en', 'speed_limit_90_ar', 'speed_limit_90_en', 'speed_limit_90', 'stop', 'traffic_light', 'u_turn']

# Créez une fonction pour effectuer la prédiction sur une image donnée
def predict(image):
    # Faire une prédiction sur l'image
    results = model(image)
    # Récupérer l'image annotée avec les boîtes englobantes
    return results



# Configurer la page avec un titre et un logo
st.set_page_config(page_title='YOLOv5 Object Detection', page_icon=':guardsman:', layout='wide')




# Créez une interface utilisateur Streamlit
def main():
    
    # Ajouter une section pour le nom du réalisateur et le titre du projet
    st.markdown(
        """
        <div style="background-color:#0072C6;padding:10px;border-radius:10px">
        <h2 style="color:white;text-align:center;">La détéction et la classification <br>des paneaux routières</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    

# Créer une zone de contenu avec un fond coloré
    bg_color = "#b3d1ff"

    # Définir la couleur de fond pour les colonnes
    col_bg_color = "#f4ff88"

    # Définir le style pour la zone principale
    page_bg_style = f"""
        <style>
            .reportview-container {{
                background-color: {bg_color};
            }}
        </style>
    """
    st.markdown(page_bg_style, unsafe_allow_html=True)

    # Styliser la sidebar
    st.sidebar.markdown("<div style='background-color: " + col_bg_color + "; padding: 0px; border-radius: 20px;text-align:center;line-height: 0.2;'>"
                    "<h2>Bienvenue sur notre interface</h2>"
                    "</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='background-color: " + bg_color + "; padding: 30px; border-radius: 30px;text-align:center;line-height:0.05;'>"
                    "<h2 style='text-decoration: underline;line-height:0.05;'>réalisé par:</h2>"
                     "<ul>"
                    "<h3>FATHLLAH Ikrame</h3>"
                    "<h3>SERIANI Boutaina</h3>"
                    "<h2 style='text-decoration: underline; line-height:0.05;'>encadré par:</h2>"
                    "<h3>Pr. ZNIYED Yassine</h3>"
                    "</div>", unsafe_allow_html=True)
    
    image_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"],)
                                          
                    
   
    # Diviser la zone principale en deux colonnes
    col1, col2 = st.columns(2)
    # Ajouter une barre verticale entre les deux colonnes
    
    with col1:
        # Ajouter une zone pour la sélection de l'image
        st.markdown("<div style='background-color: " + col_bg_color + "; padding: 10px; border-radius: 10px;text-align:center;'>"
                    "<h3>L'image choisie</h3>"
                    "</div>", unsafe_allow_html=True)
        # Charger l'image sélectionnée
        st.write('')
        if image_file is not None:
                image = Image.open(image_file)
                st.image(image, use_column_width=True)
        
    # Ajouter une zone pour l'affichage de la prédiction
    with col2:
        st.markdown("<div style='background-color: " + col_bg_color + "; padding: 10px; border-radius: 10px;text-align:center;'>"
                    "<h3>L'image prédite</h3>"
                    "<div id='prediction_output'></div>"
                    "</div>", unsafe_allow_html=True)
        # Ajouter un bouton pour effectuer la prédiction
        
        st.write('')
        # Ajouter un bouton pour effectuer la prédiction
        
                        # Faire une prédiction sur l'image
        if image_file is not None: 
                prediction = predict(image)
                            # Afficher l'image de prédiction
                st.image(prediction.render(), use_column_width=True) 
        else:
                    st.warning("Veuillez sélectionner une image avant de prédire.")
    if image_file is not None: 
        st.markdown("<div style='background-color: " + col_bg_color + "; padding: 10px; border-radius: 10px;text-align:center'>"
                    "<h3>Le résultat de la prédiction</h3>"
                    "<div id='prediction_output'></div>"
                    "</div>", unsafe_allow_html=True)
        
        class_idx = int(prediction.xyxy[0][0][-1])
        prob_pred = prediction.xyxy[0][0][-2]
                # Récupérer le nom de la classe correspondante
        class_name = names[class_idx]
        unsafe_allow_html=True, 

        last_result_container = st.empty()
        last_result_container.markdown(
                f'<div style="background-color:#D3D3D3; padding:10px; border-radius:10px"><p style="color:Black; font-size:20px">Classe prédite : {class_name}, <br>Probabilité : {prob_pred:.2f}</p></div>',
                unsafe_allow_html=True,
                )
    
    
if __name__ == '__main__':
    main()
