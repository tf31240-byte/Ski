import streamlit as st
import gpxpy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from haversine import haversine, Unit
import requests
from datetime import datetime, timedelta
import io
import json

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Ski Analytics Pro - AI Edition", 
    layout="wide", 
    page_icon="🏔️",
    initial_sidebar_state="expanded"
)

# Initialisation session state
if 'sessions' not in st.session_state:
    st.session_state.sessions = []
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'theme' not in st.session_state:
    st.session_state.theme = "Clair"

# --- STYLES CSS AMÉLIORÉS ---
def apply_theme(theme):
    if theme == "Sombre":
        return """
        <style>
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .stMetric {
                background: linear-gradient(135deg, #434343 0%, #000000 100%);
                padding: 15px;
                border-radius: 15px;
                color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            h1, h2, h3 {
                color: #ffffff !important;
            }
        </style>
        """
    elif theme == "Montagne":
        return """
        <style>
            .stApp {
                background: linear-gradient(to bottom, #e3f2fd 0%, #ffffff 100%);
            }
            .stMetric {
                background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                padding: 15px;
                border-radius: 15px;
                color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
        </style>
        """
    else:
        return """
        <style>
            .stMetric {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px;
                border-radius: 15px;
                color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .stMetric label {
                color: rgba(255,255,255,0.9) !important;
                font-weight: 600;
            }
            .stMetric [data-testid="stMetricValue"] {
                color: white !important;
                font-size: 28px !important;
            }
            .run-card {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
                margin: 10px 0;
            }
            .stProgress > div > div > div > div {
                background: linear-gradient(to right, #667eea, #764ba2);
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
        </style>
        """

st.markdown(apply_theme(st.session_state.theme), unsafe_allow_html=True)

# --- CONFIGURATION AVANCÉE ---
DIFFICULTY_THRESHOLDS = {
    'verte': {'min': 0, 'max': 12, 'color': '#009E60', 'hex': '#90EE90', 'score': 1},
    'bleue': {'min': 12, 'max': 25, 'color': '#007FFF', 'hex': '#ADD8E6', 'score': 2},
    'rouge': {'min': 25, 'max': 45, 'color': '#FF0000', 'hex': '#FFcccb', 'score': 3},
    'noire': {'min': 45, 'max': 100, 'color': '#000000', 'hex': '#D3D3D3', 'score': 4}
}

MAX_SPEED_KMH = 140
MIN_RUN_DURATION = 30
MIN_RUN_DROP = 30
SMOOTHING_WINDOW = 10

# --- FONCTIONS UTILITAIRES ---

def get_piste_difficulty(gradient):
    """Retourne la difficulté de la piste selon le gradient"""
    gradient = abs(gradient)
    
    for name, params in DIFFICULTY_THRESHOLDS.items():
        if params['min'] <= gradient < params['max']:
            return name.capitalize(), params['color'], params['score']
    
    return 'Noire', DIFFICULTY_THRESHOLDS['noire']['color'], 4

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calcule la distance entre deux points GPS en mètres"""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0
    try:
        return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)
    except:
        return 0

def smooth_data(series, window=SMOOTHING_WINDOW, method='rolling'):
    """Lisse une série de données"""
    if method == 'ewm':
        return series.ewm(span=window).mean()
    else:
        return series.rolling(window=window, min_periods=1, center=True).mean()

def detect_outliers_speed(speed_series, max_speed=MAX_SPEED_KMH):
    """Détecte et corrige les vitesses aberrantes"""
    speed_clean = speed_series.copy()
    speed_clean = speed_clean.replace([np.inf, -np.inf], np.nan)
    speed_clean[speed_clean > max_speed] = np.nan
    speed_clean[speed_clean < 0] = 0
    return speed_clean.interpolate(method='linear').fillna(0)

def fetch_osm_piste_name(lat, lon, timeout=3):
    """Interroge l'API Overpass OSM pour trouver le nom de la piste"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:{timeout}];
    (
        way(around:100, {lat}, {lon})["piste:type"="downhill"];
        way(around:100, {lat}, {lon})["aerialway"="chair_lift"];
        way(around:100, {lat}, {lon})["aerialway"="gondola"];
    );
    out tags;
    """
    
    try:
        response = requests.get(
            overpass_url, 
            params={'data': overpass_query}, 
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('elements'):
                for element in data['elements']:
                    tags = element.get('tags', {})
                    if tags.get('piste:type') == 'downhill':
                        name = tags.get('name', tags.get('piste:name', ''))
                        ref = tags.get('ref', tags.get('piste:ref', ''))
                        difficulty = tags.get('piste:difficulty', '')
                        
                        result = f"{name} {ref}".strip()
                        if difficulty:
                            result += f" ({difficulty})"
                        return result if result else "Piste non identifiée"
                        
        return "Hors-piste / Non répertorié"
    
    except requests.exceptions.Timeout:
        return "⏱️ Timeout OSM"
    except Exception as e:
        return f"❌ Erreur OSM"

def get_weather_data(lat, lon):
    """Récupère les conditions météo via Open-Meteo API"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat,
        'longitude': lon,
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum',
        'timezone': 'auto',
        'forecast_days': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        return {
            'temp_max': data['daily']['temperature_2m_max'][0],
            'temp_min': data['daily']['temperature_2m_min'][0],
            'precip': data['daily']['precipitation_sum'][0],
            'snow': data['daily']['snowfall_sum'][0]
        }
    except:
        return None

def detect_jumps(df, threshold=2):
    """Détecte les sauts basés sur l'accélération verticale"""
    df = df.copy()
    df['vertical_speed'] = df['ele_diff'] / df['dt'].replace(0, np.nan)
    df['acceleration'] = df['vertical_speed'].diff() / df['dt'].replace(0, np.nan)
    
    jumps = []
    in_air = False
    jump_start = None
    
    for idx, row in df.iterrows():
        if pd.isna(row['acceleration']):
            continue
            
        if row['acceleration'] < -threshold and not in_air:
            in_air = True
            jump_start = idx
        
        elif row['acceleration'] > threshold and in_air and jump_start is not None:
            in_air = False
            air_time = (df.loc[idx, 'time'] - df.loc[jump_start, 'time']).total_seconds()
            if air_time > 0.5 and air_time < 5:  # Entre 0.5s et 5s
                jumps.append({
                    'time': df.loc[jump_start, 'time'],
                    'air_time': round(air_time, 2),
                    'height_estimate': round((air_time**2 * 9.81) / 8, 1)
                })
    
    return pd.DataFrame(jumps)

def detect_rest_zones(df):
    """Détecte les zones où vous vous êtes arrêté"""
    df = df.copy()
    rest_zones = []
    
    df['is_stopped'] = (df['speed_kmh'] < 2) & (df['state'] == 'Ski')
    df['stop_group'] = (df['is_stopped'] != df['is_stopped'].shift()).cumsum()
    
    for group_id, group in df[df['is_stopped']].groupby('stop_group'):
        duration = group['dt'].sum()
        if duration > 10:
            rest_zones.append({
                'lat': group['lat'].mean(),
                'lon': group['lon'].mean(),
                'duration': round(duration, 1),
                'time': group['time'].iloc[0]
            })
    
    return pd.DataFrame(rest_zones)

def calculate_session_score(runs_df, df):
    """Calcule un score global de la session"""
    score = 0
    
    # Distance totale
    total_dist = df[df['state'] == 'Ski']['dist'].sum() / 1000
    score += min(total_dist * 2, 50)
    
    # Nombre de descentes
    score += min(len(runs_df) * 5, 25)
    
    # Dénivelé
    denivele = df['cumulative_descent'].max()
    score += min(denivele / 100, 25)
    
    # Variété
    variety = len(runs_df['Couleur Dominante'].unique())
    score += variety * 5
    
    return min(int(score), 100)

def get_recommendations(runs_df, df):
    """Suggère des améliorations"""
    recommendations = []
    
    avg_speed = runs_df['Vitesse Moy (km/h)'].mean()
    
    if avg_speed < 20:
        recommendations.append("💡 Essayez de maintenir une vitesse plus constante")
    
    color_counts = runs_df['Couleur Dominante'].value_counts()
    if 'Noire' not in color_counts.index:
        recommendations.append("🎯 Challenge : Tentez une piste noire !")
    
    if len(runs_df) < 5:
        recommendations.append("🔥 Vous pouvez augmenter le nombre de descentes")
    
    if avg_speed > 50:
        recommendations.append("⚠️ Attention à la vitesse, privilégiez la sécurité")
    
    # Détection fatigue
    if len(runs_df) > 3:
        last_3_avg = runs_df.tail(3)['Vitesse Moy (km/h)'].mean()
        first_3_avg = runs_df.head(3)['Vitesse Moy (km/h)'].mean()
        
        if last_3_avg < first_3_avg * 0.85:
            recommendations.append("😴 Signes de fatigue détectés, pensez à faire une pause")
    
    return recommendations

def parse_and_enrich_gpx(file_content):
    """Parse un fichier GPX et enrichit les données"""
    try:
        if isinstance(file_content, str):
            gpx = gpxpy.parse(file_content)
        else:
            gpx = gpxpy.parse(file_content)
        
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
        
        if not data:
            raise ValueError("Aucune donnée GPS trouvée dans le fichier")
        
        df = pd.DataFrame(data)
        df = df.sort_values('time').reset_index(drop=True)
        
        # Calculs de base
        df['prev_lat'] = df['lat'].shift(1)
        df['prev_lon'] = df['lon'].shift(1)
        df['prev_ele'] = df['ele'].shift(1)
        df['prev_time'] = df['time'].shift(1)
        
        df['dist'] = df.apply(
            lambda x: calculate_distance(x['lat'], x['lon'], x['prev_lat'], x['prev_lon']),
            axis=1
        )
        
        df['dt'] = (df['time'] - df['prev_time']).dt.total_seconds().fillna(0)
        
        # Vitesse
        df['speed_raw'] = np.where(
            df['dt'] > 0,
            (df['dist'] / df['dt']) * 3.6,
            0
        )
        
        df['speed_kmh'] = detect_outliers_speed(df['speed_raw'])
        df['speed_kmh'] = smooth_data(df['speed_kmh'], window=5, method='ewm')
        
        # Dénivelé et gradient
        df['ele_diff'] = df['ele'] - df['prev_ele']
        
        df['gradient'] = np.where(
            df['dist'] > 0.5,
            -(df['ele_diff'] / df['dist']) * 100,
            0
        )
        
        df['gradient'] = smooth_data(df['gradient'].clip(-100, 100), window=15)
        
        # Classification
        difficulty_data = df['gradient'].apply(get_piste_difficulty)
        df['color_name'] = difficulty_data.apply(lambda x: x[0])
        df['hex_color'] = difficulty_data.apply(lambda x: x[1])
        df['difficulty_score'] = difficulty_data.apply(lambda x: x[2])
        
        # État
        df['state'] = 'Arrêt'
        
        df.loc[
            (df['ele_diff'] > 1) & 
            (df['speed_kmh'] > 3) & 
            (df['speed_kmh'] < 30),
            'state'
        ] = 'Remontée'
        
        df.loc[
            (df['ele_diff'] < -0.5) & 
            (df['speed_kmh'] > 5),
            'state'
        ] = 'Ski'
        
        # Métriques cumulatives
        df['cumulative_dist'] = df['dist'].cumsum() / 1000
        df['cumulative_descent'] = df['ele_diff'].clip(upper=0).abs().cumsum()
        df['cumulative_ascent'] = df['ele_diff'].clip(lower=0).cumsum()
        
        df['color_rgb'] = df['hex_color'].apply(
            lambda x: [int(x[1:3], 16), int(x[3:5], 16), int(x[5:7], 16)]
        )
        
        return df
    
    except Exception as e:
        raise Exception(f"Erreur lors du parsing GPX : {str(e)}")

def detect_runs(df):
    """Détecte et analyse les descentes individuelles"""
    df['segment_id'] = (df['state'] != df['state'].shift()).cumsum()
    
    runs = []
    run_number = 1
    
    for seg_id, group in df.groupby('segment_id'):
        if group['state'].iloc[0] != 'Ski':
            continue
        
        duration = group['dt'].sum()
        drop = group['ele'].max() - group['ele'].min()
        distance = group['dist'].sum()
        
        if duration < MIN_RUN_DURATION or drop < MIN_RUN_DROP:
            continue
        
        avg_speed = group['speed_kmh'].mean()
        max_speed = group['speed_kmh'].max()
        avg_gradient = group['gradient'].abs().mean()
        max_gradient = group['gradient'].abs().max()
        
        dominant_color = group['color_name'].mode()[0] if not group['color_name'].mode().empty else "Inconnue"
        
        color_distribution = group['color_name'].value_counts(normalize=True) * 100
        
        mid_idx = len(group) // 2
        mid_point = group.iloc[mid_idx]
        
        start_point = group.iloc[0]
        end_point = group.iloc[-1]
        
        runs.append({
            'N°': run_number,
            'id': seg_id,
            'Début': group['time'].iloc[0],
            'Fin': group['time'].iloc[-1],
            'Durée (s)': int(duration),
            'Durée': f"{int(duration//60)}:{int(duration%60):02d}",
            'Dénivelé (m)': int(drop),
            'Distance (m)': int(distance),
            'Vitesse Moy (km/h)': round(avg_speed, 1),
            'Vitesse Max (km/h)': int(max_speed),
            'Pente Moy (%)': round(avg_gradient, 1),
            'Pente Max (%)': round(max_gradient, 1),
            'Couleur Dominante': dominant_color,
            'Distribution Couleurs': color_distribution.to_dict(),
            'lat_center': mid_point['lat'],
            'lon_center': mid_point['lon'],
            'lat_start': start_point['lat'],
            'lon_start': start_point['lon'],
            'lat_end': end_point['lat'],
            'lon_end': end_point['lon'],
            'ele_start': start_point['ele'],
            'ele_end': end_point['ele'],
            'points': group
        })
        
        run_number += 1
    
    return pd.DataFrame(runs)

def create_elevation_profile(df):
    """Crée un graphique du profil d'altitude avec couleurs"""
    fig = go.Figure()
    
    for color in df['color_name'].unique():
        mask = df['color_name'] == color
        color_hex = df[mask]['hex_color'].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=df[mask]['cumulative_dist'],
            y=df[mask]['ele'],
            mode='lines',
            name=color,
            line=dict(color=color_hex, width=3),
            hovertemplate='<b>Distance:</b> %{x:.2f} km<br>' +
                          '<b>Altitude:</b> %{y:.0f} m<br>' +
                          '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Profil d'Altitude par Difficulté",
        xaxis_title="Distance (km)",
        yaxis_title="Altitude (m)",
        hovermode='x unified',
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def create_speed_chart(df):
    """Graphique de vitesse dans le temps"""
    fig = px.line(
        df[df['state'] == 'Ski'],
        x='time',
        y='speed_kmh',
        title='Vitesse au fil du temps',
        labels={'time': 'Heure', 'speed_kmh': 'Vitesse (km/h)'},
        template='plotly_white'
    )
    
    fig.update_traces(line_color='#667eea', line_width=2)
    fig.update_layout(height=350)
    
    return fig

def analyze_fatigue(runs_df):
    """Détecte la baisse de performance au fil des descentes"""
    if len(runs_df) < 2:
        return None
    
    runs_df = runs_df.copy()
    runs_df['vitesse_relative'] = (
        runs_df['Vitesse Moy (km/h)'] / runs_df['Vitesse Moy (km/h)'].iloc[0]
    ) * 100
    
    fig = px.line(
        runs_df,
        x='N°',
        y='vitesse_relative',
        title='Évolution de la Performance',
        labels={'vitesse_relative': 'Performance relative (%)', 'N°': 'Numéro de descente'}
    )
    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Performance initiale")
    fig.update_layout(height=300, template='plotly_white')
    
    return fig

def create_session_score_gauge(score):
    """Crée une jauge pour le score de session"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score de Session"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "#e0e0e0"},
                {'range': [75, 90], 'color': "#90EE90"},
                {'range': [90, 100], 'color': "#FFD700"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_3d_map(df):
    """Crée une carte 3D interactive avec PyDeck"""
    ski_data = df[df['state'] == 'Ski'].copy()
    
    if ski_data.empty:
        return None
    
    # Layer de trajectoire
    path_layer = pdk.Layer(
        "PathLayer",
        data=ski_data,
        pickable=True,
        get_path=ski_data[['lon', 'lat']].values.tolist(),
        get_color='color_rgb',
        width_scale=20,
        width_min_pixels=3,
        get_width=5,
    )
    
    # Heatmap de vitesse
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=ski_data,
        get_position=['lon', 'lat'],
        get_weight='speed_kmh',
        radius_pixels=50,
        intensity=1,
        threshold=0.05,
        opacity=0.5
    )
    
    # Points de départ/arrivée
    start_end_data = pd.DataFrame([
        {'lon': ski_data.iloc[0]['lon'], 'lat': ski_data.iloc[0]['lat'], 'color': [0, 255, 0, 200], 'type': 'Départ'},
        {'lon': ski_data.iloc[-1]['lon'], 'lat': ski_data.iloc[-1]['lat'], 'color': [255, 0, 0, 200], 'type': 'Arrivée'}
    ])
    
    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        data=start_end_data,
        get_position=['lon', 'lat'],
        get_color='color',
        get_radius=50,
        pickable=True,
    )
    
    view_state = pdk.ViewState(
        latitude=ski_data['lat'].mean(),
        longitude=ski_data['lon'].mean(),
        zoom=13,
        pitch=60,
        bearing=0
    )
    
    r = pdk.Deck(
        layers=[heatmap_layer, path_layer, marker_layer],
        initial_view_state=view_state,
        tooltip={
            "text": "Vitesse: {speed_kmh:.1f} km/h\nAltitude: {ele:.0f} m\nPente: {gradient:.1f}%"
        },
        map_style="mapbox://styles/mapbox/satellite-streets-v12"
    )
    
    return r

def export_to_csv(runs_df):
    """Exporte les résultats en CSV"""
    export_df = runs_df[[
        'N°', 'Début', 'Fin', 'Durée', 'Dénivelé (m)', 
        'Distance (m)', 'Vitesse Moy (km/h)', 'Vitesse Max (km/h)',
        'Pente Moy (%)', 'Couleur Dominante'
    ]].copy()
    
    return export_df.to_csv(index=False).encode('utf-8')

def export_session_json(runs_df, df, score):
    """Exporte la session complète en JSON"""
    session_data = {
        'date': df['time'].iloc[0].isoformat(),
        'score': score,
        'stats': {
            'total_runs': len(runs_df),
            'total_descent': int(df['cumulative_descent'].max()),
            'total_distance': round(df['cumulative_dist'].max(), 2),
            'max_speed': round(df['speed_kmh'].max(), 1),
            'max_altitude': int(df['ele'].max())
        },
        'runs': runs_df[[
            'N°', 'Dénivelé (m)', 'Distance (m)', 
            'Vitesse Max (km/h)', 'Couleur Dominante'
        ]].to_dict('records')
    }
    
    return json.dumps(session_data, indent=2).encode('utf-8')

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/skiing.png", width=80)
    st.title("⚙️ Paramètres")
    
    # Thème
    st.markdown("---")
    st.subheader("🎨 Apparence")
    theme = st.selectbox(
        "Thème",
        ["Clair", "Sombre", "Montagne"],
        index=["Clair", "Sombre", "Montagne"].index(st.session_state.theme)
    )
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.rerun()
    
    st.markdown("---")
    st.subheader("🔧 Filtres de détection")
    
    min_duration = st.slider(
        "Durée minimale (s)", 
        10, 120, MIN_RUN_DURATION,
        help="Durée minimale d'une descente valide"
    )
    
    min_drop = st.slider(
        "Dénivelé minimal (m)", 
        10, 200, MIN_RUN_DROP,
        help="Dénivelé minimal d'une descente valide"
    )
    
    st.markdown("---")
    st.subheader("📊 Affichage")
    
    show_3d = st.checkbox("Carte 3D", value=True)
    show_heatmap = st.checkbox("Heatmap de vitesse", value=True)
    show_stats = st.checkbox("Statistiques détaillées", value=True)
    show_jumps = st.checkbox("Détection de sauts", value=False)
    show_weather = st.checkbox("Données météo", value=False)
    
    # Historique des sessions
    st.markdown("---")
    st.subheader("📚 Historique")
    
    if len(st.session_state.sessions) > 0:
        st.write(f"**{len(st.session_state.sessions)} session(s) enregistrée(s)**")
        
        for i, session in enumerate(st.session_state.sessions):
            with st.expander(f"Session {i+1} - {session['date'].strftime('%d/%m/%Y')}"):
                st.write(f"🏔️ Descentes: {session['runs']}")
                st.write(f"⬇️ Dénivelé: {session['denivele']:.0f} m")
                st.write(f"🚀 Vitesse max: {session['vitesse_max']:.1f} km/h")
                st.write(f"🏆 Score: {session['score']}/100")
        
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.sessions = []
            st.rerun()
    
    st.markdown("---")
    st.info("""
    **💡 Astuces :**
    - Exportez en GPX depuis Slopes
    - OSM identifie les pistes
    - Ctrl+Clic pour pivoter la 3D
    - Sauvegardez vos sessions
    """)

# --- INTERFACE PRINCIPALE ---
st.title("⛷️ Ski Analytics Pro - Édition IA")
st.markdown("""
Analysez automatiquement vos sessions de ski : détection de difficulté, 
identification des pistes, statistiques avancées et bien plus !
""")

uploaded_file = st.file​​​​​​​​​​​​​​​​
_uploader(
“📂 Importez votre fichier GPX”,
type=None,
help=“Tous formats acceptés - Le fichier sera analysé automatiquement”
)
if uploaded_file is not None:
try:
with st.spinner(“🔄 Analyse du fichier en cours…”):
file_content = uploaded_file.read()
        try:
            file_content = file_content.decode('utf-8')
        except:
            file_content = file_content.decode('latin-1')
        
        MIN_RUN_DURATION = min_duration
        MIN_RUN_DROP = min_drop
        
        df = parse_and_enrich_gpx(file_content)
        runs_df = detect_runs(df)
    
    st.success(f"✅ Fichier '{uploaded_file.name}' analysé avec succès !")
    
    # Calcul du score
    score = calculate_session_score(runs_df, df)
    
    # --- SECTION 1 : STATISTIQUES GLOBALES ---
    st.markdown("### 📊 Vue d'ensemble de la session")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("🏔️ Altitude Max", f"{int(df['ele'].max())} m")
    
    with col2:
        st.metric("⬇️ Dénivelé Total", f"{int(df['cumulative_descent'].max())} m")
    
    with col3:
        st.metric("🛷 Descentes", len(runs_df))
    
    with col4:
        st.metric("🚀 Vitesse Max", f"{df['speed_kmh'].max():.1f} km/h")
    
    with col5:
        total_ski_time = df[df['state'] == 'Ski']['dt'].sum()
        st.metric("⏱️ Temps Ski", f"{int(total_ski_time//60)} min")
    
    with col6:
        st.metric("🏆 Score", f"{score}/100")
    
    # Météo si activé
    if show_weather:
        weather = get_weather_data(df['lat'].mean(), df['lon'].mean())
        if weather:
            st.markdown("---")
            wcol1, wcol2, wcol3 = st.columns(3)
            with wcol1:
                st.markdown(f"🌡️ **Température:** {weather['temp_min']}°C - {weather['temp_max']}°C")
            with wcol2:
                st.markdown(f"❄️ **Neige:** {weather['snow']} cm")
            with wcol3:
                st.markdown(f"🌧️ **Précipitations:** {weather['precip']} mm")
    
    st.markdown("---")
    
    # --- SECTION 2 : GRAPHIQUES ---
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.plotly_chart(
            create_elevation_profile(df[df['state'] == 'Ski']),
            use_container_width=True
        )
    
    with col_right:
        st.plotly_chart(
            create_speed_chart(df),
            use_container_width=True
        )
    
    # Score et fatigue
    score_col, fatigue_col = st.columns(2)
    
    with score_col:
        st.plotly_chart(
            create_session_score_gauge(score),
            use_container_width=True
        )
    
    with fatigue_col:
        fatigue_chart = analyze_fatigue(runs_df)
        if fatigue_chart:
            st.plotly_chart(fatigue_chart, use_container_width=True)
    
    # Sauts si activé
    if show_jumps and not df.empty:
        jumps_df = detect_jumps(df[df['state'] == 'Ski'])
        if not jumps_df.empty:
            st.markdown("### 🪂 Sauts Détectés")
            jump_col1, jump_col2 = st.columns([1, 2])
            with jump_col1:
                st.metric("Nombre de sauts", len(jumps_df))
                st.metric("Temps en l'air total", f"{jumps_df['air_time'].sum():.1f} s")
            with jump_col2:
                st.dataframe(
                    jumps_df.style.format({
                        'air_time': '{:.2f} s',
                        'height_estimate': '{:.1f} m'
                    }),
                    use_container_width=True
                )
    
    # --- SECTION 3 : TABLEAU DES DESCENTES ---
    st.markdown("### 🎿 Détail des Descentes")
    
    if not runs_df.empty:
        col_osm, col_export, col_save = st.columns([2, 1, 1])
        
        with col_osm:
            identify_names = st.checkbox(
                "🔍 Identifier les pistes via OpenStreetMap",
                help="Interroge OSM (peut prendre du temps)"
            )
        
        with col_export:
            csv_data = export_to_csv(runs_df)
            st.download_button(
                label="📥 CSV",
                data=csv_data,
                file_name=f"ski_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col_save:
            json_data = export_session_json(runs_df, df, score)
            st.download_button(
                label="💾 JSON",
                data=json_data,
                file_name=f"session_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        # Sauvegarder la session
        if st.button("💾 Sauvegarder cette session dans l'historique"):
            st.session_state.sessions.append({
                'date': df['time'].iloc[0],
                'runs': len(runs_df),
                'denivele': df['cumulative_descent'].max(),
                'vitesse_max': df['speed_kmh'].max(),
                'score': score
            })
            st.success("✅ Session sauvegardée !")
            st.rerun()
        
        # Identification OSM
        if identify_names:
            with st.spinner("🌍 Interrogation d'OpenStreetMap..."):
                progress_bar = st.progress(0)
                names = []
                
                for i, row in runs_df.iterrows():
                    name = fetch_osm_piste_name(row['lat_center'], row['lon_center'])
                    names.append(name)
                    progress_bar.progress((i + 1) / len(runs_df))
                
                runs_df['Nom Piste (OSM)'] = names
                st.success("✅ Identification terminée !")
        else:
            runs_df['Nom Piste (OSM)'] = "Non recherché"
        
        def color_difficulty(val):
            colors = {
                'Verte': 'background-color: #90EE90; color: black; font-weight: bold',
                'Bleue': 'background-color: #ADD8E6; color: black; font-weight: bold',
                'Rouge': 'background-color: #FFcccb; color: black; font-weight: bold',
                'Noire': 'background-color: #D3D3D3; color: black; font-weight: bold'
            }
            return colors.get(val, '')
        
        display_cols = [
            'N°', 'Début', 'Nom Piste (OSM)', 'Couleur Dominante',
            'Dénivelé (m)', 'Distance (m)', 'Durée', 
            'Vitesse Max (km/h)', 'Pente Max (%)'
        ]
        
        styled_df = runs_df[display_cols].style.applymap(
            color_difficulty,
            subset=['Couleur Dominante']
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Statistiques détaillées
        if show_stats:
            st.markdown("### 📈 Statistiques Avancées")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.markdown("**Distribution des Pistes**")
                color_counts = runs_df['Couleur Dominante'].value_counts()
                for color, count in color_counts.items():
                    st.write(f"{color} : {count} descente(s)")
            
            with stat_col2:
                st.markdown("**Moyennes**")
                st.write(f"Dénivelé moyen : {runs_df['Dénivelé (m)'].mean():.0f} m")
                st.write(f"Vitesse moyenne : {runs_df['Vitesse Moy (km/h)'].mean():.1f} km/h")
                st.write(f"Durée moyenne : {runs_df['Durée (s)'].mean()//60:.0f} min")
            
            with stat_col3:
                st.markdown("**Records**")
                st.write(f"Plus longue : {runs_df['Distance (m)'].max():.0f} m")
                st.write(f"Plus rapide : {runs_df['Vitesse Max (km/h)'].max():.0f} km/h")
                st.write(f"Plus raide : {runs_df['Pente Max (%)'].max():.1f} %")
            
            # Zones de repos
            rest_df = detect_rest_zones(df)
            if not rest_df.empty:
                st.markdown("**🛑 Zones de repos**")
                st.write(f"{len(rest_df)} arrêt(s) détecté(s) (>{10}s)")
    
    # Recommandations
    st.markdown("### 💡 Recommandations Personnalisées")
    recommendations = get_recommendations(runs_df, df)
    for rec in recommendations:
        st.info(rec)
    
    # --- SECTION 4 : CARTE 3D ---
    if show_3d:
        st.markdown("### 🗺️ Visualisation 3D Interactive")
        
        deck = create_3d_map(df)
        if deck:
            st.pydeck_chart(deck)
            st.info("💡 Maintenez Ctrl + Clic pour faire pivoter la vue en 3D")
        else:
            st.warning("Aucune donnée de descente à afficher")
    
    # Détails par descente
    with st.expander("🔍 Détails par descente"):
        selected_run = st.selectbox(
            "Choisir une descente",
            runs_df['N°'].tolist(),
            format_func=lambda x: f"Descente #{x} - {runs_df[runs_df['N°']==x]['Couleur Dominante'].iloc[0]}"
        )
        
        if selected_run:
            run_data = runs_df[runs_df['N°'] == selected_run].iloc[0]
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown(f"""
                **Informations générales**
                - 🏷️ Nom : {run_data['Nom Piste (OSM)']}
                - 🎨 Difficulté : {run_data['Couleur Dominante']}
                - 📏 Distance : {run_data['Distance (m)']} m
                - ⬇️ Dénivelé : {run_data['Dénivelé (m)']} m
                - ⏱️ Durée : {run_data['Durée']}
                """)
            
            with detail_col2:
                st.markdown(f"""
                **Performances**
                - 🚀 Vitesse max : {run_data['Vitesse Max (km/h)']} km/h
                - 📊 Vitesse moy : {run_data['Vitesse Moy (km/h)']} km/h
                - 📐 Pente max : {run_data['Pente Max (%)']} %
                - 📐 Pente moy : {run_data['Pente Moy (%)']} %
                """)
            
            run_points = run_data['points']
            
            fig_run = go.Figure()
            fig_run.add_trace(go.Scatter(
                x=run_points['time'],
                y=run_points['speed_kmh'],
                name='Vitesse',
                line=dict(color='#667eea', width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.3)'
            ))
            
            fig_run.update_layout(
                title=f"Vitesse - Descente #{selected_run}",
                xaxis_title="Temps",
                yaxis_title="Vitesse (km/h)",
                height=300,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_run, use_container_width=True)

except Exception as e:
    st.error("❌ Erreur lors de l'analyse du fichier")
    st.exception(e)
    st.info("""
    **Vérifications :**
    - Le fichier est-il bien au format GPX ?
    - Contient-il des données de traces GPS ?
    - Essayez de réexporter depuis votre application
    """)
else:
st.info(”””
### 🎯 Comment utiliser cette application ?
1. **Exportez** vos traces depuis votre app de ski (Slopes, Ski Tracks, etc.) en GPX
2. **Importez** le fichier via le bouton ci-dessus
3. **Analysez** automatiquement :
   - Difficulté des pistes (Verte/Bleue/Rouge/Noire)
   - Vitesses, dénivelés et statistiques
   - Identification des pistes via OpenStreetMap
   - Détection de sauts et zones de repos
   - Score de session et recommandations
   - Visualisation 3D interactive

### ✨ Nouvelles Fonctionnalités
- 🪂 **Détection de sauts** avec estimation de hauteur
- 📊 **Analyse de fatigue** sur les descentes
- 🏆 **Score de session** personnalisé
- 💡 **Recommandations** basées sur vos performances
- 📚 **Historique** de vos sessions
- 🌡️ **Données météo** en temps réel
- 🎨 **Thèmes** personnalisables
- 🗺️ **Heatmap** de vitesse sur carte 3D
""")

st.markdown("### 📊 Exemple de résultat")

example_data = pd.DataFrame({
    'N°': [1, 2, 3],
    'Piste': ['Les Crêtes', 'Bellecôte', 'Face Nord'],
    'Couleur': ['Rouge', 'Bleue', 'Noire'],
    'Dénivelé': [450, 320, 580],
    'Vitesse Max': [78, 62, 85],
    'Durée': ['5:23', '4:12', '6:45']
})

st.dataframe(example_data, use_container_width=True)
Footer
st.markdown(”—”)
st.markdown(”””
<div style='text-align: center; color: #666;'>
    <p>Ski Analytics Pro v3.0 - Édition Complète | Streamlit × OpenStreetMap × Open-Meteo</p>
    <p>🏔️ Analysez, Comparez, Progressez, Dépassez-vous ! 🏔️</p>
</div>
""", unsafe_allow_html=True)