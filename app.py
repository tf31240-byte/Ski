import streamlit as st
import gpxpy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from haversine import haversine, Unit
import math
import requests



# --- CONFIGURATION ---
st.set_page_config(page_title="Ski Analytics V2 - AI Edition", layout="wide", page_icon="🏔️")

st.markdown("""
<style>
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
    .css-1r6slb0 {border: 1px solid #ddd; border-radius: 5px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS UTILITAIRES ---

def get_piste_difficulty(gradient):
    """Retourne la couleur et le code HEX selon la pente (%)"""
    # Standards approximatifs : Vert < 15%, Bleu < 25%, Rouge < 40%, Noir > 40%
    if gradient < 12: return 'Verte', '#009E60', 1
    elif gradient < 25: return 'Bleue', '#007FFF', 2
    elif gradient < 45: return 'Rouge', '#FF0000', 3
    else: return 'Noire', '#000000', 4

def fetch_osm_piste_name(lat, lon):
    """Interroge l'API Overpass pour trouver le nom de la piste proche"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    way(around:50, {lat}, {lon})["piste:type"="downhill"];
    out tags;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=2)
        data = response.json()
        if len(data['elements']) > 0:
            tags = data['elements'][0]['tags']
            name = tags.get('name', 'Piste Inconnue')
            ref = tags.get('ref', '')
            return f"{name} {ref}".strip()
    except:
        return None
    return "Hors-Piste / Inconnu"

def parse_and_enrich_gpx(file):
    gpx = gpxpy.parse(file)
    data = []
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append({
                    'time': point.time,
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation
                })
    
    df = pd.DataFrame(data)
    
    # 1. Calculs de base
    df['prev_lat'] = df['lat'].shift(1)
    df['prev_lon'] = df['lon'].shift(1)
    df['prev_ele'] = df['ele'].shift(1)
    df['prev_time'] = df['time'].shift(1)
    
    # Distance & Temps
    df['dist'] = df.apply(lambda x: haversine((x['lat'], x['lon']), (x['prev_lat'], x['prev_lon']), unit=Unit.METERS) if pd.notnull(x['prev_lat']) else 0, axis=1)
    df['dt'] = (df['time'] - df['prev_time']).dt.total_seconds().fillna(0)
    
    # Vitesse (Lissée)
    df['speed_kmh'] = (df['dist'] / df['dt']) * 3.6
    df['speed_kmh'] = df['speed_kmh'].replace([np.inf, -np.inf], 0).fillna(0)
    df.loc[df['speed_kmh'] > 140, 'speed_kmh'] = np.nan # Filtre erreur GPS
    df['speed_kmh'] = df['speed_kmh'].interpolate().fillna(0)

    # 2. Calcul de la Pente (Gradient) pour la COULEUR
    # Pente % = (Dénivelé / Distance) * 100
    df['ele_diff'] = df['ele'] - df['prev_ele']
    df['gradient'] = np.where(df['dist'] > 1, -(df['ele_diff'] / df['dist']) * 100, 0)
    df['gradient'] = df['gradient'].rolling(window=10).mean().fillna(0) # Lissage fort
    
    # Classification Couleur
    df['color_name'], df['hex_color'], df['difficulty_score'] = zip(*df['gradient'].apply(get_piste_difficulty))
    
    # 3. Détection Descente vs Remontée
    # Si on monte (ele_diff positif) et vitesse > 2 m/s -> Remontée
    # Si on descend (ele_diff negatif) et vitesse > 2 m/s -> Ski
    df['state'] = 'Attente'
    df.loc[(df['ele_diff'] > 0.5) & (df['speed_kmh'] > 5), 'state'] = 'Remontée'
    df.loc[(df['ele_diff'] < -0.5) & (df['speed_kmh'] > 5), 'state'] = 'Ski'
    
    return df

def detect_runs(df):
    """Groupe les points en 'Runs' distincts"""
    df['segment_id'] = (df['state'] != df['state'].shift()).cumsum()
    
    runs = []
    for seg_id, group in df.groupby('segment_id'):
        if group['state'].iloc[0] == 'Ski':
            duration = group['dt'].sum()
            drop = group['ele'].max() - group['ele'].min()
            
            if duration > 60 and drop > 50:
                # Analyse de la couleur dominante
                dominant_color = group['color_name'].mode()[0]
                
                # Coordonnées pour l'API OSM (milieu de piste)
                mid_point = group.iloc[len(group)//2]
                
                runs.append({
                    'id': seg_id,
                    'Début': group['time'].iloc[0],
                    'Durée': f"{int(duration//60)}m {int(duration%60)}s",
                    'Dénivelé': int(drop),
                    'Vitesse Max': int(group['speed_kmh'].max()),
                    'Couleur Estimée': dominant_color,
                    'Pente Max': int(group['gradient'].max()),
                    'lat_center': mid_point['lat'],
                    'lon_center': mid_point['lon'],
                    'points': group
                })
    return pd.DataFrame(runs)

# --- INTERFACE ---

st.title("⛷️ Ski Analytics V2 : Reconnaissance IA")
st.markdown("Analyse automatique de la difficulté des pistes et visualisation

uploaded_file = st.file_uploader("📂 Importez votre fichier (GPX, TXT, XML...)", type=None)

if uploaded_file is not None:
    try:
        # On lit le contenu du fichier
        file_content = uploaded_file.read().decode("utf-8")
        
        # On tente de parser le contenu même si l'extension est bizarre
        try:
            gpx = gpxpy.parse(file_content)
            
            if not gpx.tracks:
                st.error("Le fichier a été lu, mais il ne semble pas contenir de traces GPS valides.")
            else:
                st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès !")
                # Appelez ici votre fonction de traitement habituelle :
                # df = parse_and_enrich_gpx_from_string(file_content)
                # ...
                
        except Exception as e:
            st.error("Erreur de lecture : Ce fichier n'est pas au format GPX (même s'il est débloqué).")
            st.info("Astuce : Assurez-vous d'avoir bien exporté depuis Slopes en format 'GPX'.")

    except Exception as e:
        st.error(f"Impossible de lire ce fichier : {e}")
        
        # --- SECTION 1 : STATS GLOBALES ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Vitesse Max", f"{df['speed_kmh'].max():.1f} km/h")
        col2.metric("Distance Totale", f"{df['dist'].sum()/1000:.1f} km")
        col3.metric("Descentes", len(runs_df))
        col4.metric("Altitude Max", f"{int(df['ele'].max())} m")
        
        # --- SECTION 2 : RECONNAISSANCE DES PISTES ---
        st.subheader("🔎 Analyse des Descentes (Couleur & Nom)")
        
        if not runs_df.empty:
            # Bouton pour l'IA OSM
            identify_names = st.checkbox("🔍 Interroger OpenStreetMap pour trouver le nom des pistes (peut être lent)")
            
            if identify_names:
                progress_bar = st.progress(0)
                names = []
                for i, row in runs_df.iterrows():
                    name = fetch_osm_piste_name(row['lat_center'], row['lon_center'])
                    names.append(name)
                    progress_bar.progress((i + 1) / len(runs_df))
                runs_df['Nom Détecté (OSM)'] = names
                st.success("Identification terminée !")
            else:
                runs_df['Nom Détecté (OSM)'] = "Non recherché"

            # Affichage du Tableau Coloré
            def color_highlight(val):
                color = 'white'
                if val == 'Verte': color = '#90EE90' # LightGreen
                elif val == 'Bleue': color = '#ADD8E6' # LightBlue
                elif val == 'Rouge': color = '#FFcccb' # LightRed
                elif val == 'Noire': color = '#D3D3D3' # Grey
                return f'background-color: {color}; color: black'

            st.dataframe(
                runs_df[['Début', 'Nom Détecté (OSM)', 'Couleur Estimée', 'Dénivelé', 'Vitesse Max', 'Durée']]
                .style.applymap(color_highlight, subset=['Couleur Estimée']),
                use_container_width=True
            )
            
            # --- SECTION 3 : VISUALISATION 3D ---
            st.subheader("⛰️ Visualisation 3D du Relief")
            
            # Préparation des données pour PyDeck
            # On colore chaque point selon sa difficulté
            df['color_rgb'] = df['hex_color'].apply(lambda x: [int(x[1:3], 16), int(x[3:5], 16), int(x[5:7], 16)])
            
            layer = pdk.Layer(
                "PathLayer",
                data=df[df['state'] == 'Ski'], # On n'affiche que le ski
                pickable=True,
                get_path="[[lon, lat]]", # PyDeck demande [lon, lat]
                get_color="color_rgb",
                width_scale=20,
                width_min_pixels=2,
                get_width=5,
            )
            
            # Vue initiale centrée
            view_state = pdk.ViewState(
                latitude=df['lat'].mean(),
                longitude=df['lon'].mean(),
                zoom=12,
                pitch=50, # Inclinaison 3D
            )
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Vitesse: {speed_kmh} km/h"},
                map_style="mapbox://styles/mapbox/satellite-v9"
            )
            
            st.pydeck_chart(r)
            
            st.info("💡 Astuce : Maintenez le clic droit (ou Ctrl + Clic) sur la carte pour faire pivoter la vue en 3D.")

        else:
            st.warning("Aucune descente détectée.")