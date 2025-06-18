import streamlit as st
from PIL import Image
import os
import folium
from streamlit_folium import st_folium
from huggingface_hub import hf_hub_download # NOUVEL IMPORT

# --- NEW IMPORTS FOR CHATBOT FUNCTIONALITY ---
from groq import Groq
from dotenv import load_dotenv
import base64
import tempfile

# --- NOUVEAUX IMPORTS POUR LA FONCTIONNALITÉ D'ANALYSE DE PLANTE ---
import tensorflow as tf
import numpy as np

# --- Configuration de la page Streamlit ---
# st.set_page_config() DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT EXÉCUTÉE
st.set_page_config(
    page_title="GreenField Pro - Dashboard",
    page_icon=":seedling:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM AUDIO RECORDER COMPONENT PLACEHOLDER ---
# Keep the declaration for now, in case it's used elsewhere or for future re-addition,
# but its usage is removed from show_chatbot_view.
try:
    import streamlit.components.v1 as components
    # ASSUREZ-VOUS QUE CE CHEMIN EST CORRECT PAR RAPPORT À LA RACINE DE VOTRE DÉPÔT
    _audio_recorder_component = components.declare_component(
        "audio_recorder_component",
        path="./audio_recorder_component/frontend/build" # Adjust this path to your component's build folder
    )
    def audio_recorder_component():
        return _audio_recorder_component(key="audio_recorder_widget")
except Exception as e:
    st.warning(f"Could not load custom audio recorder component. Audio recording functionality might be limited: {e}")
    def audio_recorder_component():
        st.info("Audio recording component not loaded. Please ensure it's installed correctly and ffmpeg/ffprobe are available.")
        return None

# --- Chemins des ressources statiques et du modèle ---
LOGO_PATH = os.path.join("static", "images", "Green Plant and Agriculture Logo (2).png")
CSS_PATH = "style.css"

# Hugging Face Model Repository et Nom de Fichier du Modèle
HF_REPO_ID = "mopaoleonel/plante_tedection" # Votre nom d'utilisateur/repo
HF_MODEL_FILENAME = "mon_modele.keras"    # Le nom exact du fichier modèle dans votre repo HF

# --- Dictionnaire des classes de maladies ---
CLASS_NAMES = {
    'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3,
    'Blueberry___healthy': 4, 'Cherry_(including_sour)___Powdery_mildew': 5, 'Cherry_(including_sour)___healthy': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8, 'Corn_(maize)___Northern_Leaf_Blight': 9,
    'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11, 'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
    'Grape___healthy': 14, 'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17,
    'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20, 'Potato___Late_blight': 21,
    'Potato___healthy': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24, 'Squash___Powdery_mildew': 25,
    'Strawberry___Leaf_scorch': 26, 'Strawberry___healthy': 27, 'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29,
    'Tomato___Late_blight': 30, 'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32, 'Tomato___Spider_mites Two-spotted_spider_mite': 33,
    'Tomato___Target_Spot': 34, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato___Tomato_mosaic_virus': 36, 'Tomato___healthy': 37
}
# Inverser le dictionnaire pour un accès facile par index
CLASS_NAMES_INV = {v: k for k, v in CLASS_NAMES.items()}

# Taille d'entrée attendue par votre modèle
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3 # RGB

# --- Fonctions utilitaires et de chargement ---

def load_css(css_file):
    """Charge un fichier CSS externe."""
    try:
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Fichier CSS non trouvé à : {css_file}. L'interface utilisera les styles par défaut.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du CSS : {e}")

@st.cache_resource # Met en cache le modèle pour éviter les rechargements coûteux
def load_keras_model_from_hub(repo_id, filename):
    """Télécharge et charge le modèle Keras depuis Hugging Face Hub."""
    try:
        # Télécharge le fichier du modèle dans le cache de Hugging Face (et Streamlit)
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        st.info(f"Modèle téléchargé depuis Hugging Face: {model_path}")

        # Charge le modèle TensorFlow
        model = tf.keras.models.load_model(model_path)
        st.success("Modèle Keras chargé avec succès !")
        return model
    except Exception as e:
        st.error(f"Erreur Critique : Impossible de télécharger ou charger le modèle depuis Hugging Face Hub: {e}")
        st.info(f"Assurez-vous que le dépôt '{repo_id}' et le fichier '{filename}' existent et sont accessibles. Vérifiez également que votre modèle a été sauvegardé avec TensorFlow {tf.__version__}.")
        return None

# --- Fonctions pour les vues spécifiques ---

def show_dashboard_view():
    """Affiche le tableau de bord principal avec les métriques et la carte."""
    st.markdown("<h1>Tableau de Bord</h1>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <a href="#" class="info-card">
                <div>
                    <div class="value">24°C</div>
                    <div class="label">Température</div>
                </div>
                <div class="icon-container"><i class="fas fa-thermometer-three-quarters"></i></div>
            </a>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <a href="#" class="info-card">
                <div>
                    <div class="value">42.5%</div>
                    <div class="label">Humidité</div>
                </div>
                <div class="icon-container"><i class="fas fa-tint"></i></div>
            </a>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <a href="#" class="info-card">
                <div>
                    <div class="value">3.0mm</div>
                    <div class="label">Précipitation</div>
                </div>
                <div class="icon-container"><i class="fas fa-cloud-rain"></i></div>
            </a>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <a href="#" class="info-card">
                <div>
                    <div class="value">3.5m/s</div>
                    <div class="label">Vent</div>
                </div>
                <div class="icon-container"><i class="fas fa-wind"></i></div>
            </a>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<h2 id='map-plantation'><i class='fa-solid fa-map-location-dot'></i> Carte des Plantations</h2>", unsafe_allow_html=True)

    center_lat = 5.4746
    center_lon = 10.4243

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)

    folium.Marker(
        location=[center_lat, center_lon],
        popup="<b>Bafoussam</b><br>Zone de Plantation Principale",
        tooltip="Bafoussam"
    ).add_to(m)

    folium.Marker(
        location=[5.62, 10.05],
        popup="<b>Zone Ouest</b><br>Quelques fermes ici.",
        icon=folium.Icon(color="green", icon="leaf")
    ).add_to(m)

    st_folium(m, width='100%', height=500)

    st.markdown("""
        <p style='color: var(--text-color); margin-top: 1rem;'>
            Visualisation des emplacements de vos plantations et des points d'intérêt.
        </p>
    """, unsafe_allow_html=True)


def show_plant_analysis_view(groq_client, model):
    """Affiche la page d'analyse de maladie des plantes."""
    st.markdown("<h2 id='analyse-plant'><i class='fa-solid fa-microscope'></i> Analyser ma plante</h2>", unsafe_allow_html=True)
    st.markdown("Téléchargez une image de la feuille de votre plante pour obtenir un diagnostic instantané et des recommandations.")

    if not model:
        st.error("Le service d'analyse est indisponible car le modèle de prédiction n'a pas pu être chargé. Veuillez vérifier la connexion à Hugging Face et le nom du modèle.")
        return

    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="plant_analysis_uploader")

    if uploaded_file is not None:
        col1, col2 = st.columns([0.8, 1.2]) # Donner plus de place aux résultats
        with col1:
            st.image(uploaded_file, caption="Image téléchargée", use_container_width=True)

        with col2:
            with st.spinner("Analyse de l'image en cours..."):
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    # REDIMENSIONNEMENT À LA TAILLE ATTENDUE PAR VOTRE MODÈLE
                    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = tf.expand_dims(img_array, 0) # Ajoute une dimension de batch

                    # NORMALISATION: Assurez-vous que c'est la même que celle utilisée lors de l'entraînement
                    # Si votre modèle attend des valeurs entre 0 et 1
                    img_array = tf.cast(img_array, tf.float32) / 255.0
                    # Si votre modèle gère la normalisation en interne (ex: Rescaling layer), supprimez la ligne ci-dessus.

                    prediction = model.predict(img_array)
                    predicted_class_index = np.argmax(prediction)
                    predicted_class_name = CLASS_NAMES_INV[predicted_class_index]

                    # Formatage du nom pour une meilleure lisibilité
                    parts = predicted_class_name.split('___')
                    plant_name = parts[0].replace('_', ' ')
                    disease_name = parts[1].replace('_', ' ')

                except Exception as e:
                    st.error(f"Erreur lors du traitement de l'image ou de la prédiction : {e}")
                    st.info("Assurez-vous que l'image est valide et que le modèle est compatible.")
                    return # Arrête le traitement si erreur

            st.subheader("Résultats de l'analyse")
            if 'healthy' in disease_name.lower():
                st.success(f"**Diagnostic :** La plante ({plant_name}) semble **saine**.")
                st.balloons()
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader("Conseils pour une plante saine")
                if not groq_client:
                    st.info("Le service de conseils est indisponible car la clé API Groq n'a pas été configurée.")
                    st.markdown("<p style='font-style: italic; color: #888;'>Veuillez configurer votre clé API Groq pour obtenir des conseils détaillés.</p>", unsafe_allow_html=True)
                else:
                    with st.spinner(f"Génération de conseils pour maintenir {plant_name} saine..."):
                        try:
                            prompt = f"""
                            Je suis un agriculteur. Mon application a diagnostiqué que ma plante '{plant_name}' est saine.
                            Donne-moi des conseils pratiques et concis pour maintenir la bonne santé de cette plante.
                            Structure ta réponse comme suit en Markdown :

                            ### 🌱 Conseils pour une Plante Saine
                            - [Conseil 1]
                            - [Conseil 2]
                            - [Etc...]

                            Ajoute une petite conclusion encourageante.
                            """
                            chat_completion = groq_client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model="llama3-70b-8192", # Utilisez le bon modèle Groq
                            )
                            response_text = chat_completion.choices[0].message.content
                            st.markdown(response_text)
                        except Exception as e:
                            st.error(f"Impossible de générer les conseils. Erreur API Groq : {e}. Veuillez vérifier votre connexion et votre clé API.")

            else:
                st.warning(f"**Diagnostic :** {plant_name} - **{disease_name}**")

                if not groq_client:
                    st.info("Le service de recommandation est indisponible car la clé API Groq n'a pas été configurée.")
                    st.markdown("<p style='font-style: italic; color: #888;'>Veuillez configurer votre clé API Groq pour obtenir des recommandations détaillées.</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    with st.spinner(f"Recherche des causes et solutions pour '{disease_name}'..."):
                        try:
                            prompt = f"""
                            Je suis un agriculteur et mon application a diagnostiqué la maladie '{disease_name}' sur ma plante '{plant_name}'.
                            Fournis-moi une fiche d'information claire et concise en français. Structure ta réponse exactement comme suit, en utilisant Markdown :

                            ### 🧐 Causes Principales
                            - [Cause 1]
                            - [Cause 2]
                            - [Etc...]

                            ### 🛡️ Solutions et Traitements
                            **Prévention :**
                            - [Mesure préventive 1]
                            - [Mesure préventive 2]
                            **Traitements Biologiques :**
                            - [Traitement biologique 1]
                            **Traitements Chimiques :**
                            - [Traitement chimique 1]

                            Ajoute à la fin une petite conclusion encourageante.
                            """
                            chat_completion = groq_client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model="llama3-70b-8192", # Utilisez le bon modèle Groq
                            )
                            response_text = chat_completion.choices[0].message.content
                            st.markdown(response_text)
                        except Exception as e:
                            st.error(f"Impossible de générer les recommandations. Erreur API Groq : {e}. Veuillez vérifier votre connexion et votre clé API.")


def show_chatbot_view(groq_client):
    """Affiche la page de l'assistant virtuel (chatbot)."""
    st.markdown("<h2><i class='fa-solid fa-comments'></i> Assistant Virtuel</h2>", unsafe_allow_html=True)
    st.markdown("Posez-moi n'importe quelle question sur l'agriculture, vos cultures, ou les résultats de vos analyses.")

    if not groq_client:
        st.warning("Veuillez configurer votre clé API Groq pour utiliser l'assistant virtuel.")
        st.info("Ajoutez `GROQ_API_KEY=\"votre_cle_ici\"` dans un fichier `.env` à la racine de votre projet.")
        return

    # Bouton pour effacer l'historique dans la barre latérale
    with st.sidebar:
        if st.button("Effacer l'historique du Chat", use_container_width=True, key="clear_chat_button"):
            st.session_state.messages = [{"role": "bot", "content": "Bonjour ! Je suis votre assistant virtuel GreenField. Comment puis-je vous aider aujourd'hui ?"}]
            st.rerun()

    # Initialisation de l'historique du chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "bot", "content": "Bonjour ! Je suis votre assistant virtuel GreenField. Comment puis-je vous aider aujourd'hui ?"}]

    # Affichage des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar='🧑‍🌾' if message["role"] == 'user' else '🤖'):
            st.markdown(message["content"])

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # --- Form for text input and file uploader ---
    with st.form("chat_input_form", clear_on_submit=True):
        message_input = st.text_input("💬 Entrez votre message ici :", "", key="chat_text_input", autocomplete="off")
        audio_file_uploader = st.file_uploader("📢 Téléversez un message audio (format m4a, mp3, wav)", type=["m4a", "mp3", "wav"], key="chat_audio_uploader")
        send_button = st.form_submit_button("✉️ Envoyer Message", type="primary")

    processed_message_content = ""

    # Logique de traitement du formulaire soumis (fichier audio uploadé ou texte)
    if send_button: # Now this is the primary entry point for sending messages
        if audio_file_uploader:
            st.info("Traitement du fichier audio téléversé...")
            filename = None # Initialize filename
            try:
                filename = tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file_uploader.name.split('.')[-1]}").name
                with open(filename, "wb") as f:
                    f.write(audio_file_uploader.getvalue())

                with open(filename, "rb") as file_to_transcribe:
                    with st.spinner("Transcription audio en cours..."):
                        transcription = groq_client.audio.transcriptions.create( # Utilisez groq_client ici
                            file=(audio_file_uploader.name, file_to_transcribe.read()),
                            model="whisper-large-v3",
                            response_format="json",
                            language="fr",
                            temperature=0.0
                        )
                processed_message_content = transcription.text
                st.session_state.messages.append({"role": "user", "content": processed_message_content})
            except Exception as e:
                st.error(f"Erreur Transcription (Fichier Téléversé): Impossible de transcrire le fichier audio. Vérifiez le format et votre clé API Groq. Erreur: {e}")
                processed_message_content = ""
            finally:
                if filename and os.path.exists(filename): # Clean up temp file
                    os.remove(filename)
        elif message_input:
            processed_message_content = message_input
            st.session_state.messages.append({"role": "user", "content": processed_message_content})

        # Si un message a été traité avec succès (audio uploadé ou texte)
        if processed_message_content:
            with st.spinner("Le chatbot réfléchit..."):
                try:
                    chat_completion = groq_client.chat.completions.create( # Utilisez groq_client ici
                        messages=[{"role": "user", "content": processed_message_content}],
                        model="llama3-70b-8192", # Mettez le modèle Groq que vous utilisez
                    )
                    response_text = chat_completion.choices[0].message.content
                    st.session_state.messages.append({"role": "bot", "content": response_text})

                except Exception as e:
                    st.error(f"Erreur Chatbot (API): Impossible d'obtenir une réponse du chatbot Groq. Erreur: {e}")
                    st.session_state.messages.append({"role": "bot", "content": "Désolé, je n'ai pas pu traiter votre demande. Une erreur est survenue lors de la communication avec le service de chatbot."})

        st.rerun() # Force rerun to update chat history and show response


# --- LOGIQUE PRINCIPALE DE L'APPLICATION ---

# Chargement du CSS et Font Awesome
load_css(CSS_PATH)
st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">""", unsafe_allow_html=True)

# Initialisation du client Groq (gestion des erreurs et clés)
groq_client = None
try:
    load_dotenv() # Charge les variables d'environnement du fichier .env
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
    else:
        st.sidebar.warning("GROQ_API_KEY non trouvée. Le chatbot et les recommandations d'analyse ne seront pas disponibles. Veuillez l'ajouter à votre fichier `.env`.")
except Exception as e:
    st.sidebar.error(f"Erreur d'initialisation Groq : {e}. Le chatbot et les recommandations seront désactivés.")

# Chargement du modèle Keras depuis Hugging Face (mise en cache automatique par @st.cache_resource)
keras_model = load_keras_model_from_hub(HF_REPO_ID, HF_MODEL_FILENAME)

# Configuration de la barre latérale
with st.sidebar:
    try:
        logo_image = Image.open(LOGO_PATH)
        st.image(logo_image, use_container_width=True)
    except FileNotFoundError:
        st.warning(f"Logo non trouvé à {LOGO_PATH}. L'image par défaut sera utilisée.")
    except Exception as e:
        st.warning(f"Erreur lors du chargement du logo : {e}")

    st.markdown("<h2 style='text-align: center;'>Menu Principal</h2>", unsafe_allow_html=True) # Titre du menu

    # Boutons de navigation
    if st.button("📊 Tableau de Bord", use_container_width=True, key="nav_dashboard"):
        st.session_state.current_view = "dashboard"
        st.rerun()

    # Bouton pour la nouvelle vue d'analyse de plante
    if st.button("🌱 Analyser ma plante", use_container_width=True, key="nav_analysis"):
        st.session_state.current_view = "plant_analysis" # Nouvelle vue
        st.rerun()

    if st.button("💬 Assistant Virtuel", use_container_width=True, key="nav_chat"):
        st.session_state.current_view = "chat"
        st.rerun()

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True) # Séparateur

    # Bouton de déconnexion (simulé)
    if st.button("🚪 Déconnexion", use_container_width=True, key="nav_logout"):
        st.warning("Déconnexion simulée. Ajoutez votre logique de déconnexion ici.")
        st.session_state.current_view = "dashboard" # Redirige vers le tableau de bord après déconnexion
        st.rerun() # Pour rafraîchir la page après la déconnexion simulée

    st.markdown("<div style='text-align: center; font-size: 0.8rem; color: var(--text-color-light); margin-top: 2rem;'>© 2025 GreenField Pro. Tous droits réservés.</div>", unsafe_allow_html=True)


# Logique de routage pour afficher la bonne vue
if "current_view" not in st.session_state:
    st.session_state.current_view = "dashboard" # Vue par défaut

if st.session_state.current_view == "dashboard":
    show_dashboard_view()
elif st.session_state.current_view == "plant_analysis": # Appel à la nouvelle vue
    show_plant_analysis_view(groq_client, keras_model)
elif st.session_state.current_view == "chat":
    show_chatbot_view(groq_client)
else: # Fallback pour toute valeur inattendue
    show_dashboard_view()
