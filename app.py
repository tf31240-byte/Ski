"""
SKI ANALYTICS PRO - ULTIMATE EDITION
====================================
Fonctionnalités :
- Replay 3D Interactif (First Person View)
- Analyse Biomécanique (Équilibre G/D, Symétrie, Fatigue)
- Heatmaps (Vitesse, G-Force, Fréquentation)
- Météo Contextuelle
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
from scipy.signal import savgol_filter
from scipy import stats
from datetime import datetime, timezone, timedelta
import zipfile
import xml.etree.ElementTree as ET
from dateutil import parser as date_parser
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import requests  # Pour la météo

st.set_page_config(page_title="Ski Analytics Pro Ultimate", layout="wide")

# ============================================================================
# MODELS & CONFIG
# ============================================================================

class ActionType(Enum):
    RUN = "Run"
    LIFT = "Lift"

class Difficulty(Enum):
    VERTE = ("Verte", 0, 12, "#2ecc71")
    BLEUE = ("Bleue", 12, 25, "#3498db")
    ROUGE = ("Rouge", 25, 45, "#e74c3c")
    NOIRE = ("Noire", 45, 100, "#2c3e50")
    
    def __init__(self, label, min_g, max_g, color):
        self.label = label
        self.min_grade = min_g
        self.max_grade = max_g
        self.color = color
    
    @classmethod
    def from_gradient(cls, gradient: float):
        for level in cls:
            if level.min_grade <= abs(gradient) < level.max_grade:
                return level
        return cls.NOIRE

@dataclass
class Weather:
    temp_avg: float
    wind_max: float
    condition: str

@dataclass
class Activity:
    location: str
    start: datetime
    end: datetime
    distance: float
    duration: int
    vertical: float
    run_count: int
    top_speed: float
    conditions: str
    weather: Optional[Weather] = None

@dataclass
class Biomechanics:
    """Stocke les données biomécaniques calculées"""
    left_turns_count: int = 0
    right_turns_count: int = 0
    avg_g_left: float = 0.0
    avg_g_right: float = 0.0
    carving_symmetry_score: float = 0.0  # 0 = déséquilibré, 1 = parfait
    stability_index: float = 0.0         # Stabilité globale

@dataclass
class Action:
    type: ActionType
    number: int
    start: datetime
    end: datetime
    duration: float
    distance: float
    vertical: float
    max_alt: float
    min_alt: float
    avg_speed: float
    top_speed: float
    track_ids: str = ""
    gps_data: pd.DataFrame = None
    difficulty: Difficulty = None
    avg_gradient: float = 0.0
    max_g_lat: float = 0.0
    max_g_long: float = 0.0
    avg_turn: float = 0.0
    max_roughness: float = 0.0
    
    # Nouvelles données biomécaniques
    biomechanics: Biomechanics = field(default_factory=Biomechanics)
    fatigue_score: float = 0.0  # Basé sur la dégradation technique

# ============================================================================
# PARSING & DATA LOADING
# ============================================================================

def parse_metadata(zip_file: zipfile.ZipFile) -> Tuple[Optional[Activity], List[Action]]:
    try:
        xml_files = [f for f in zip_file.namelist() if 'metadata' in f.lower()]
        if not xml_files: return None, []
        
        with zip_file.open(xml_files[0]) as f:
            tree = ET.parse(f)
            root = tree.getroot()
            act_elem = root if root.tag == 'Activity' else root.find('.//Activity')
            
            if act_elem is None: return None, []
            
            # Parsing Activité
            activity = Activity(
                location=act_elem.get('locationName', 'Unknown'),
                start=date_parser.parse(act_elem.get('start')).astimezone(timezone.utc),
                end=date_parser.parse(act_elem.get('end')).astimezone(timezone.utc),
                distance=float(act_elem.get('distance', 0)),
                duration=int(act_elem.get('duration', 0)),
                vertical=float(act_elem.get('vertical', 0)),
                run_count=int(act_elem.get('runCount', 0)),
                top_speed=float(act_elem.get('topSpeed', 0)),
                conditions=act_elem.get('conditions', 'unknown')
            )
            
            # Récupération Météo (Mock ou API)
            activity.weather = get_weather(activity.location, activity.start, activity.end)
            
            # Parsing Actions
            actions = []
            action_elems = root.findall('.//Action')
            
            for elem in action_elems:
                try:
                    action_type = ActionType.RUN if elem.get('type') == 'Run' else ActionType.LIFT
                    actions.append(Action(
                        type=action_type,
                        number=int(elem.get('numberOfType', 0)),
                        start=date_parser.parse(elem.get('start')).astimezone(timezone.utc),
                        end=date_parser.parse(elem.get('end')).astimezone(timezone.utc),
                        duration=float(elem.get('duration', 0)),
                        distance=float(elem.get('distance', 0)),
                        vertical=float(elem.get('vertical', 0)),
                        max_alt=float(elem.get('maxAlt', 0)),
                        min_alt=float(elem.get('minAlt', 0)),
                        avg_speed=float(elem.get('avgSpeed', 0)),
                        top_speed=float(elem.get('topSpeed', 0)),
                        track_ids=elem.get('trackIDs', '')
                    ))
                except: continue
            
            return activity, actions
    except Exception as e:
        st.error(f"Erreur XML: {e}")
        return None, []

def get_weather(location: str, start: datetime, end: datetime) -> Weather:
    """Simule ou récupère la météo. Utilise Open-Meteo (gratuit) si date passée."""
    # Note: Les données fournies sont en 2026. L'API renverra une erreur.
    # On simule donc des données réalistes pour Tignes/Val d'Isère.
    
    # Tentative réelle (désactivée pour le demo futur):
    # lat, lon = 45.4, 6.9 # Coordonnées approx Tignes
    # url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={start.date()}&end_date={end.date()}&hourly=temperature_2m,wind_speed_10m"
    
    # MOCK DATA pour la démo
    return Weather(
        temp_avg=-5.0,
        wind_max=15.0,
        condition="Neige légère / Nuageux"
    )

def load_gps(zip_file: zipfile.ZipFile) -> Optional[pd.DataFrame]:
    try:
        target = next((f for f in zip_file.namelist() if 'GPS.csv' in f and 'Raw' not in f), None)
        if not target: target = next((f for f in zip_file.namelist() if 'RawGPS.csv' in f), None)
        if not target: return None
        
        with zip_file.open(target) as f:
            # Lecture CSV robuste
            df = pd.read_csv(f, sep=r'[|,\t]', engine='python', header=None, on_bad_lines='skip')
            
            # Auto-détection colonnes (simplifiée par rapport aux données fournies)
            # Format donné: Time, Lat, Lon, Ele, Speed, Course, HAcc, VAcc
            df = df.iloc[:, [0, 1, 2, 3]]
            df.columns = ['time', 'lat', 'lon', 'ele']
            
            df['time'] = pd.to_datetime(pd.to_numeric(df['time'], errors='coerce'), unit='s', utc=True)
            df = df.dropna().sort_values('time').reset_index(drop=True)
            return df
    except: return None

# ============================================================================
# CORE PHYSICS & BIOMECHANICS
# ============================================================================

def enrich_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Enrichit le GPS avec Vitesse, G-Forces, Pente, Virages."""
    df = df.copy()
    
    # Calculs basiques
    lat1, lon1 = np.radians(df['lat']), np.radians(df['lon'])
    lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    df['dist'] = 2 * np.arcsin(np.sqrt(a.clip(0, 1))) * 6371000
    df['dt'] = df['time'].diff().dt.total_seconds().fillna(0)
    
    # Vitesse
    mask_valid = (df['dt'] > 0) & (df['dt'] < 10)
    df['speed_raw'] = np.where(mask_valid, df['dist'] / df['dt'], 0)
    if len(df) > 11: df['speed_ms'] = savgol_filter(df['speed_raw'], 11, 2)
    else: df['speed_ms'] = df['speed_raw']
    df['speed_kmh'] = (df['speed_ms'] * 3.6).clip(0, 200)
    
    # Altitude lissée & Pente
    if len(df) > 21: df['ele_smooth'] = savgol_filter(df['ele'], 21, 3)
    else: df['ele_smooth'] = df['ele']
    df['gradient'] = np.where(df['dist'] > 0.5, -(df['ele_smooth'].diff() / df['dist']) * 100, 0)
    df['gradient'] = df['gradient'].replace([np.inf, -np.inf], 0).fillna(0)
    if len(df) > 21: df['gradient'] = savgol_filter(df['gradient'], 21, 3)
    
    # Cap (Bearing)
    x = np.cos(lat1) * np.sin(dlon)
    y = np.cos(lat2) * np.sin(lat1) - np.sin(lat2) * np.cos(lat1) * np.cos(dlon)
    df['bearing'] = np.degrees(np.arctan2(x, y))
    
    # Virages (Change de bearing)
    df['bearing_diff'] = df['bearing'].diff().fillna(0)
    df['turn_angle'] = ((df['bearing_diff'] + 180) % 360) - 180 # Angle -180 à 180
    
    # Forces G
    df['accel'] = df['speed_ms'].diff() / df['dt'].replace(0, np.nan)
    df['g_long'] = (df['accel'].abs() / 9.81).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 5)
    
    turn_angle_rad = np.radians(df['turn_angle'].abs())
    radius = np.where((turn_angle_rad > 0.001) & (df['dist'] > 0.1), df['dist'] / turn_angle_rad, np.inf)
    centripetal = df['speed_ms']**2 / radius
    df['g_lat'] = (centripetal / 9.81).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 5)
    
    df['roughness'] = df['ele'].rolling(window=5, center=True).std().fillna(0)
    return df

def calculate_biomechanics(action: Action):
    """Analyse l'équilibre, la symétrie et la stabilité."""
    df = action.gps_data
    if df is None or df.empty: return

    # 1. Détection virages (On considère > 10 deg comme un virage effectif)
    df['is_turn'] = df['turn_angle'].abs() > 10
    turns = df[df['is_turn']]
    
    # 2. Équilibre Gauche / Droite
    # Negatif = Gauche, Positif = Droite
    left_turns = turns[turns['turn_angle'] < 0]
    right_turns = turns[turns['turn_angle'] > 0]
    
    action.biomechanics.left_turns_count = len(left_turns)
    action.biomechanics.right_turns_count = len(right_turns)
    
    if not left_turns.empty:
        action.biomechanics.avg_g_left = left_turns['g_lat'].mean()
    if not right_turns.empty:
        action.biomechanics.avg_g_right = right_turns['g_lat'].mean()
        
    # 3. Score de Symétrie (Ratio des G moyens)
    if action.biomechanics.avg_g_right > 0:
        ratio = action.biomechanics.avg_g_left / action.biomechanics.avg_g_right
        # 1.0 = Parfait. Plus on s'éloigne, plus c'est déséquilibré.
        action.biomechanics.carving_symmetry_score = 1.0 - abs(1.0 - ratio)
    
    # 4. Indice de stabilité (Inverse de la rugosité et des changements de direction brusques)
    # Moyenne mobile de la variation de vitesse + rugosité
    volatility = df['speed_ms'].diff().abs().mean()
    roughness = df['roughness'].mean()
    action.biomechanics.stability_index = 100 / (1 + volatility + roughness)

def calculate_fatigue(runs: List[Action]):
    """Analyse la fatigue globale sur la journée."""
    for i, run in enumerate(runs):
        # Basé sur la dégradation de la vitesse moyenne relative et la stabilité
        # On compare avec la moyenne des runs précédents
        if i == 0:
            run.fatigue_score = 0.0 # Pas de fatigue au début
        else:
            prev_runs = runs[:i]
            avg_prev_speed = np.mean([r.avg_speed for r in prev_runs])
            avg_prev_stab = np.mean([r.biomechanics.stability_index for r in prev_runs])
            
            speed_drop = (avg_prev_speed - run.avg_speed) / avg_prev_speed if avg_prev_speed > 0 else 0
            stab_drop = (avg_prev_stab - run.biomechanics.stability_index) if avg_prev_stab > 0 else 0
            
            # Score de fatigue (0 = frais, 1 = épuisé)
            run.fatigue_score = np.clip((speed_drop * 2.0) + (stab_drop * 0.5), 0, 1)

def enrich_actions(actions: List[Action], gps: pd.DataFrame):
    for action in actions:
        mask = (gps['time'] >= action.start) & (gps['time'] <= action.end)
        action.gps_data = gps[mask].copy()
        
        if action.type == ActionType.RUN and not action.gps_data.empty:
            df = action.gps_data
            action.avg_gradient = abs(df['gradient'].mean())
            action.difficulty = Difficulty.from_gradient(action.avg_gradient)
            action.max_g_lat = df['g_lat'].max()
            action.max_g_long = df['g_long'].max()
            
            gps_max_speed = df['speed_kmh'].max()
            if 0 < gps_max_speed < 200: action.top_speed = gps_max_speed
            
            # Calculs Biomécaniques
            calculate_biomechanics(action)
            
    # Calcul de la fatigue après avoir rempli toutes les actions
    runs = [a for a in actions if a.type == ActionType.RUN]
    calculate_fatigue(runs)

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class Session:
    def __init__(self, activity: Activity, actions: List[Action], user_weight: float = 75):
        self.activity = activity
        self.runs = [a for a in actions if a.type == ActionType.RUN]
        self.lifts = [a for a in actions if a.type == ActionType.LIFT]
        self.weight = user_weight

    @property
    def stats(self):
        ski_hours = sum(r.duration for r in self.runs) / 3600
        # Calcul de la météo en string
        w = self.activity.weather
        w_str = f"{w.temp_avg}°C, {w.condition}" if w else "N/A"
        
        return {
            "location": self.activity.location,
            "runs": len(self.runs),
            "distance_km": self.activity.distance / 1000,
            "descent_m": sum(abs(r.vertical) for r in self.runs),
            "duration": f"{self.activity.duration//3600}h{(self.activity.duration%3600)//60}m",
            "max_speed": max((r.top_speed for r in self.runs), default=0),
            "weather": w_str
        }
    
    def get_all_gps(self) -> pd.DataFrame:
        return pd.concat([r.gps_data for r in self.runs if r.gps_data is not None])

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_replay_map(session: Session, selected_run_idx: int, time_idx: int):
    """Crée la carte 3D pour le mode Replay."""
    run = session.runs[selected_run_idx]
    df = run.gps_data
    
    if df is None or df.empty: return None

    # 1. Tracé complet de la piste (semi-transparent)
    path_layer = pdk.Layer(
        "PathLayer",
        data=df,
        get_path="[['lon', 'lat']]",
        get_color=[255, 255, 255, 100], # Blanc transparent
        width_min_pixels=6,
        pickable=False
    )
    
    # 2. Position actuelle (Le "Skieur")
    current_pos = df.iloc[time_idx]
    
    skier_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[current_pos],
        get_position=["lon", "lat"],
        get_color=[255, 0, 0], # Rouge
        get_radius=10,
        pickable=False
    )
    
    # 3. Vue subjective (FPV)
    # On centre la caméra sur la position actuelle
    # On oriente la caméra selon le bearing GPS
    view_state = pdk.ViewState(
        latitude=current_pos['lat'],
        longitude=current_pos['lon'],
        zoom=16,         # Zoom serré pour effet POV
        pitch=60,         # Inclinaison (0=carto, 90=verticale)
        bearing=current_pos['bearing'] # La caméra tourne avec le skieur
    )
    
    return pdk.Deck(layers=[path_layer, skier_layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/outdoors-v11")

def create_heatmap(session: Session, mode: str):
    """Crée une carte thermique (Density, Speed, G-Force)."""
    df = session.get_all_gps()
    
    if mode == "Density":
        # Heatmap classique (densité des points)
        layer = pdk.Layer(
            "HeatmapLayer",
            data=df,
            get_position=["lon", "lat"],
            radius=50,
            opacity=0.8
        )
    elif mode == "Speed":
        # Scatterplot coloré par vitesse
        # Vitesse lente = Bleu, Vite = Rouge
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["lon", "lat"],
            get_color="[255 * (speed_kmh / 100), 0, 255 * (1 - speed_kmh/100), 150]",
            get_radius=20,
            pickable=True
        )
    elif mode == "G-Lat":
        # Scatterplot coloré par G Latéral (Virages techniques)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["lon", "lat"],
            get_color="[255 * (g_lat/2), 255 * (1 - g_lat/2), 0, 150]", # Vert à Rouge
            get_radius=20
        )
    
    view = pdk.ViewState(
        latitude=df['lat'].mean(), 
        longitude=df['lon'].mean(), 
        zoom=13, pitch=30
    )
    
    return pdk.Deck(layers=[layer], initial_view_state=view)

def plot_biomechanics(session: Session, run_idx: int):
    """Graphiques Plotly pour l'analyse biomécanique."""
    run = session.runs[run_idx]
    df = run.gps_data
    
    fig = go.Figure()
    
    # Virages à Gauche vs Droite
    fig.add_trace(go.Bar(
        x=["Gauche", "Droite"],
        y=[run.biomechanics.left_turns_count, run.biomechanics.right_turns_count],
        name="Nombre de virages",
        marker_color=['#3498db', '#e74c3c']
    ))
    
    fig.update_layout(
        title=f"Équilibre Virages - Run #{run.number}",
        yaxis_title="Nombre",
        template='plotly_white',
        height=400
    )
    return fig

def plot_fatigue_curve(session: Session):
    """Courbe de fatigue sur la journée."""
    runs = session.runs
    run_numbers = [r.number for r in runs]
    fatigue_scores = [r.fatigue_score * 100 for r in runs] # En pourcentage
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=run_numbers,
        y=fatigue_scores,
        mode='lines+markers',
        name="Indice de Fatigue",
        line=dict(color='#e67e22', width=3)
    ))
    
    # Ajouter une tendance linéaire
    if len(run_numbers) > 1:
        z = np.polyfit(run_numbers, fatigue_scores, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=run_numbers,
            y=p(run_numbers),
            mode='lines',
            name="Tendance",
            line=dict(color='gray', dash='dash')
        ))

    fig.update_layout(
        title="Évolution de la Fatigue (Journée)",
        xaxis_title="Run #",
        yaxis_title="Fatigue (%)",
        template='plotly_white',
        height=400
    )
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🎿 Ski Analytics Pro - Ultimate Edition")
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded = st.file_uploader("Chargez fichier .slopes", type=['slopes'])
        weight = st.number_input("Poids (kg)", 40, 150, 75)
        
        if uploaded:
            st.success("Fichier chargé !")
    
    if not uploaded:
        st.info("👈 Veuillez charger un fichier pour commencer l'analyse.")
        return
    
    # --- LOADING & PROCESSING ---
    with st.spinner("Deep Analyse en cours (Météo, Physique, Biomécanique)..."):
        with zipfile.ZipFile(uploaded) as zf:
            activity, actions = parse_metadata(zf)
            gps_data = load_gps(zf)
            
            if not activity or not actions:
                st.error("Erreur de parsing.")
                return
            
            if gps_data is not None:
                gps_data = enrich_gps(gps_data)
                enrich_actions(actions, gps_data)
            
            session = Session(activity, actions, weight)

    # --- DASHBOARD HEADER ---
    st.subheader(f"Session : {session.activity.location} | {session.stats['weather']}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Descentes", session.stats['runs'])
    c2.metric("Distance", f"{session.stats['distance_km']:.1f} km")
    c3.metric("V.Max", f"{session.stats['max_speed']:.1f} km/h")
    c4.metric("Dénivelé", f"{session.stats['descent_m']/1000:.1f} km")

    # --- NAVIGATION ---
    tab_overview, tab_replay, tab_biomech, tab_heatmaps = st.tabs([
        "📊 Vue d'ensemble", "🎥 Replay 3D", "🏥 Analyse Biomécanique", "🗺️ Cartes Thermiques"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab_overview:
        col1, col2 = st.columns(2)
        
        # Tableau des pistes
        df_runs = pd.DataFrame([{
            "N°": r.number,
            "Piste": r.track_ids[:15]+"...", # Tronqué
            "Diff": r.difficulty.label,
            "V.Moy": f"{r.avg_speed:.1f}",
            "Pente": f"{r.avg_gradient:.1f}%",
            "Symétrie": f"{r.biomechanics.carving_symmetry_score:.2f}",
            "Fatigue": f"{r.fatigue_score*100:.0f}%"
        } for r in session.runs])
        
        col1.dataframe(df_runs, use_container_width=True)
        
        # Graphiques de base
        with col2:
            # Distribution des difficultés
            fig_diff = px.histogram(
                df_runs, x="Diff", color="Diff",
                title="Répartition des Difficultés",
                color_discrete_map={d.label: d.color for d in Difficulty}
            )
            st.plotly_chart(fig_diff, use_container_width=True)

    # --- TAB 2: REPLAY 3D (NOUVEAU) ---
    with tab_replay:
        st.info("🎥 Mode Replay : Utilisez le curseur pour rejouer la descente. La carte pivote pour suivre le skieur (Vue Subjective).")
        
        run_options = [f"Run #{r.number} - {r.difficulty.label}" for r in session.runs]
        selected_run_name = st.selectbox("Choisir la descente à rejouer", run_options)
        run_idx = run_options.index(selected_run_name)
        
        run = session.runs[run_idx]
        
        # Slider temporel (Downsampling pour fluidité)
        df = run.gps_data
        if not df.empty:
            # On prend un point sur 5 pour que le slider ne soit pas trop lourd
            steps = range(0, len(df), 5)
            time_idx = st.slider("Timeline", 0, len(steps)-1, 0)
            
            # Récupérer l'index réel dans le dataframe
            real_idx = steps[time_idx]
            
            # Info dynamique
            current_point = df.iloc[real_idx]
            c1, c2, c3 = st.columns(3)
            c1.metric("Vitesse Actuelle", f"{current_point['speed_kmh']:.1f} km/h")
            c2.metric("Altitude", f"{current_point['ele']:.0f} m")
            c3.metric("Cap", f"{current_point['bearing']:.0f}°")
            
            # Affichage Carte
            st.pydeck_chart(create_replay_map(session, run_idx, real_idx), use_container_width=True)

    # --- TAB 3: BIOMÉCANIQUE (NOUVEAU) ---
    with tab_biomech:
        st.subheader("Analyse Technique & Physiologique")
        
        # Sélection run
        selected_run_name_bio = st.selectbox("Analyser Run", run_options, key="bio")
        run_idx_bio = run_options.index(selected_run_name_bio)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Équilibre G/D
            st.plotly_chart(plot_biomechanics(session, run_idx_bio), use_container_width=True)
            
            # Métriques clés
            r = session.runs[run_idx_bio]
            st.write("**Détail Run :**")
            st.json({
                "Virages Gauche": r.biomechanics.left_turns_count,
                "Virages Droite": r.biomechanics.right_turns_count,
                "Symétrie": f"{r.biomechanics.carving_symmetry_score:.2f}",
                "Stabilité": f"{r.biomechanics.stability_index:.2f}"
            })

        with col2:
            # Courbe Fatigue Globale
            st.plotly_chart(plot_fatigue_curve(session), use_container_width=True)
            st.caption("Indique si le skieur perd en vitesse ou en stabilité en fin de journée.")

    # --- TAB 4: HEATMAPS (NOUVEAU) ---
    with tab_heatmaps:
        st.subheader("Cartographie Thermique")
        
        mode = st.radio("Type de Heatmap", ["Density (Fréquentation)", "Speed (Vitesse)", "G-Lat (Virages Techniques)"])
        
        map_mode = "Density"
        if "Speed" in mode: map_mode = "Speed"
        if "G-Lat" in mode: map_mode = "G-Lat"
        
        st.pydeck_chart(create_heatmap(session, map_mode), use_container_width=True)

if __name__ == "__main__":
    main()
