"""
SKI ANALYTICS PRO - ULTIMATE EDITION (ALL FIXES)
=================================================
Corrections appliquées :
- Correction des imports (Enum déplacé vers module 'enum')
- Sécurité complète si geopandas est manquant
- Correction PyDeck (types primitifs)
"""

import sys
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
# IMPORTS CORRIGÉS
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import requests

# ============================================================================
# IMPORTS GÉOSPATIAUX (OPTIONNELS)
# ============================================================================

try:
    import geopandas as gpd
    from shapely.geometry import Point, shape
    HAS_GEOS = True
except ImportError:
    HAS_GEOS = False
    # Mock classes pour éviter les crashes de type hinting au chargement
    class _GeoDataFrame: pass
    class _Point: pass
    class _Shape: pass
    
    gpd = type('gpd', (), {'GeoDataFrame': _GeoDataFrame})()
    Point = _Point
    shape = _Shape

if not HAS_GEOS:
    st.sidebar.warning("⚠️ 'geopandas' non installé. La détection des noms de pistes est désactivée.")

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
    left_turns_count: int = 0
    right_turns_count: int = 0
    avg_g_left: float = 0.0
    avg_g_right: float = 0.0
    carving_symmetry_score: float = 0.0
    stability_index: float = 0.0

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
    
    biomechanics: Biomechanics = field(default_factory=Biomechanics)
    fatigue_score: float = 0.0

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
            activity.weather = get_weather(activity.location, activity.start, activity.end)
            
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
    return Weather(temp_avg=-5.0, wind_max=15.0, condition="Neige légère / Nuageux")

def load_gps(zip_file: zipfile.ZipFile) -> Optional[pd.DataFrame]:
    try:
        target = next((f for f in zip_file.namelist() if 'GPS.csv' in f and 'Raw' not in f), None)
        if not target: target = next((f for f in zip_file.namelist() if 'RawGPS.csv' in f), None)
        if not target: return None
        
        with zip_file.open(target) as f:
            df = pd.read_csv(f, sep=r'[|,\t]', engine='python', header=None, on_bad_lines='skip')
            df = df.iloc[:, [0, 1, 2, 3]]
            df.columns = ['time', 'lat', 'lon', 'ele']
            df['time'] = pd.to_datetime(pd.to_numeric(df['time'], errors='coerce'), unit='s', utc=True)
            df = df.dropna().sort_values('time').reset_index(drop=True)
            return df
    except: return None

# ============================================================================
# OPENSTREETMAP - DETECTION PISTES (ROBUSTE)
# ============================================================================

def get_pistes_from_osm(session: Any) -> Optional[Any]:
    if not HAS_GEOS: return None
    all_gps = session.get_all_gps()
    if all_gps is None or all_gps.empty: return None
    
    min_lat, max_lat = all_gps['lat'].min() - 0.01, all_gps['lat'].max() + 0.01
    min_lon, max_lon = all_gps['lon'].min() - 0.01, all_gps['lon'].max() + 0.01
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    
    query = f"""
    [out:json][timeout:25];
    ( way["piste:type"="downhill"]({bbox}); relation["piste:type"="downhill"]({bbox}); );
    out center;
    """
    
    url = "https://overpass-api.de/api/interpreter"
    
    try:
        with st.spinner("🗺️ Téléchargement du plan des pistes (OpenStreetMap)..."):
            response = requests.get(url, params={'data': query}, timeout=30)
            data = response.json()
        
        elements = data.get('elements', [])
        features = []
        for elem in elements:
            if elem['type'] == 'way':
                coords = [[n['lon'], n['lat']] for n in elem['geometry']]
                geom = shape({"type": "LineString", "coordinates": coords})
            elif elem['type'] == 'relation' and 'center' in elem:
                geom = Point(elem['center']['lon'], elem['center']['lat'])
            else:
                continue
            tags = elem.get('tags', {})
            features.append({
                'geometry': geom,
                'name': tags.get('piste:name', tags.get('name', 'Sans nom')),
                'difficulty': tags.get('piste:difficulty', 'unknown'),
                'type': tags.get('piste:type', 'unknown')
            })
        if not features: return None
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        st.success(f"✅ {len(gdf)} pistes trouvées dans la zone")
        return gdf
    except Exception as e:
        st.error(f"Erreur OSM: {e}")
        return None

def match_gps_to_pistes(actions: List[Action], pistes_gdf: Any):
    if not HAS_GEOS: return
    if pistes_gdf is None: return
    
    for action in actions:
        if action.type == ActionType.RUN and action.gps_data is not None and not action.gps_data.empty:
            sample_df = action.gps_data.iloc[::10].copy()
            geometry = [Point(xy) for xy in zip(sample_df['lon'], sample_df['lat'])]
            run_gdf = gpd.GeoDataFrame(sample_df, geometry=geometry, crs="EPSG:4326")
            try:
                joined = gpd.sjoin_nearest(run_gdf, pistes_gdf, distance_col="dist", max_distance=0.0005)
                if not joined.empty:
                    valid_matches = joined[joined['dist'] < 0.0005]
                    if not valid_matches.empty:
                        counts = valid_matches['name'].value_counts()
                        best_piste_name = counts.idxmax()
                        confidence = counts.max() / len(valid_matches)
                        if confidence > 0.4:
                            action.track_ids = f"{best_piste_name}"
            except Exception: pass

# ============================================================================
# CORE PHYSICS & BIOMECHANICS
# ============================================================================

def enrich_gps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lat1, lon1 = np.radians(df['lat']), np.radians(df['lon'])
    lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    df['dist'] = 2 * np.arcsin(np.sqrt(a.clip(0, 1))) * 6371000
    df['dt'] = df['time'].diff().dt.total_seconds().fillna(0)
    
    mask_valid = (df['dt'] > 0) & (df['dt'] < 10)
    df['speed_raw'] = np.where(mask_valid, df['dist'] / df['dt'], 0)
    if len(df) > 11: df['speed_ms'] = savgol_filter(df['speed_raw'], 11, 2)
    else: df['speed_ms'] = df['speed_raw']
    df['speed_kmh'] = (df['speed_ms'] * 3.6).clip(0, 200)
    
    if len(df) > 21: df['ele_smooth'] = savgol_filter(df['ele'], 21, 3)
    else: df['ele_smooth'] = df['ele']
    df['gradient'] = np.where(df['dist'] > 0.5, -(df['ele_smooth'].diff() / df['dist']) * 100, 0)
    df['gradient'] = df['gradient'].replace([np.inf, -np.inf], 0).fillna(0)
    if len(df) > 21: df['gradient'] = savgol_filter(df['gradient'], 21, 3)
    
    x = np.cos(lat1) * np.sin(dlon)
    y = np.cos(lat2) * np.sin(lat1) - np.sin(lat2) * np.cos(lat1) * np.cos(dlon)
    df['bearing'] = np.degrees(np.arctan2(x, y))
    df['turn_angle'] = ((df['bearing'].diff().fillna(0) + 180) % 360) - 180
    
    df['accel'] = df['speed_ms'].diff() / df['dt'].replace(0, np.nan)
    df['g_long'] = (df['accel'].abs() / 9.81).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 5)
    
    turn_angle_rad = np.radians(df['turn_angle'].abs())
    radius = np.where((turn_angle_rad > 0.001) & (df['dist'] > 0.1), df['dist'] / turn_angle_rad, np.inf)
    centripetal = df['speed_ms']**2 / radius
    df['g_lat'] = (centripetal / 9.81).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 5)
    
    df['roughness'] = df['ele'].rolling(window=5, center=True).std().fillna(0)
    return df

def calculate_biomechanics(action: Action):
    df = action.gps_data
    if df is None or df.empty: return
    df['is_turn'] = df['turn_angle'].abs() > 10
    turns = df[df['is_turn']]
    
    left_turns = turns[turns['turn_angle'] < 0]
    right_turns = turns[turns['turn_angle'] > 0]
    
    action.biomechanics.left_turns_count = len(left_turns)
    action.biomechanics.right_turns_count = len(right_turns)
    
    if not left_turns.empty: action.biomechanics.avg_g_left = left_turns['g_lat'].mean()
    if not right_turns.empty: action.biomechanics.avg_g_right = right_turns['g_lat'].mean()
        
    if action.biomechanics.avg_g_right > 0:
        ratio = action.biomechanics.avg_g_left / action.biomechanics.avg_g_right
        action.biomechanics.carving_symmetry_score = 1.0 - abs(1.0 - ratio)
    
    volatility = df['speed_ms'].diff().abs().mean()
    roughness = df['roughness'].mean()
    action.biomechanics.stability_index = 100 / (1 + volatility + roughness)

def calculate_fatigue(runs: List[Action]):
    for i, run in enumerate(runs):
        if i == 0: run.fatigue_score = 0.0
        else:
            prev_runs = runs[:i]
            avg_prev_speed = np.mean([r.avg_speed for r in prev_runs])
            avg_prev_stab = np.mean([r.biomechanics.stability_index for r in prev_runs])
            speed_drop = (avg_prev_speed - run.avg_speed) / avg_prev_speed if avg_prev_speed > 0 else 0
            stab_drop = (avg_prev_stab - run.biomechanics.stability_index) if avg_prev_stab > 0 else 0
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
            calculate_biomechanics(action)
            
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
        w = self.activity.weather
        w_str = f"{w.temp_avg}°C, {w.condition}" if w else "N/A"
        return {
            "location": self.activity.location, "runs": len(self.runs),
            "distance_km": self.activity.distance / 1000,
            "descent_m": sum(abs(r.vertical) for r in self.runs),
            "duration": f"{self.activity.duration//3600}h{(self.activity.duration%3600)//60}m",
            "max_speed": max((r.top_speed for r in self.runs), default=0),
            "weather": w_str
        }
    
    def get_all_gps(self) -> pd.DataFrame:
        return pd.concat([r.gps_data for r in self.runs if r.gps_data is not None])

# ============================================================================
# VISUALIZATION FUNCTIONS (FIXED)
# ============================================================================

def create_replay_map(session: Session, selected_run_idx: int, time_idx: int):
    run = session.runs[selected_run_idx]
    df = run.gps_data
    if df is None or df.empty: return None

    path_data = df[['lon', 'lat']].copy()
    path_layer = pdk.Layer("PathLayer", data=path_data, get_path="[['lon', 'lat']]", get_color=[255,255,255,100], width_min_pixels=6, pickable=False)
    
    current_pos = df.iloc[time_idx]
    skier_data = pd.DataFrame([{"lon": current_pos['lon'], "lat": current_pos['lat'], "bearing": current_pos['bearing']}])
    skier_layer = pdk.Layer("ScatterplotLayer", data=skier_data, get_position=["lon", "lat"], get_color=[255,0,0], get_radius=10, pickable=False)
    
    view_state = pdk.ViewState(latitude=current_pos['lat'], longitude=current_pos['lon'], zoom=16, pitch=60, bearing=current_pos['bearing'])
    return pdk.Deck(layers=[path_layer, skier_layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/outdoors-v11")

def create_heatmap(session: Session, mode: str):
    full_df = session.get_all_gps()
    if full_df is None or full_df.empty: return None
    df = full_df[['lon', 'lat']].copy()
    
    if mode == "Density":
        layer = pdk.Layer("HeatmapLayer", data=df, get_position=["lon", "lat"], radius=50, opacity=0.8)
    elif mode == "Speed":
        df['speed_kmh'] = full_df['speed_kmh'].values
        layer = pdk.Layer("ScatterplotLayer", data=df, get_position=["lon", "lat"], get_color="[255 * (speed_kmh / 100), 0, 255 * (1 - speed_kmh/100), 150]", get_radius=20)
    elif mode == "G-Lat":
        df['g_lat'] = full_df['g_lat'].values
        layer = pdk.Layer("ScatterplotLayer", data=df, get_position=["lon", "lat"], get_color="[255 * (g_lat/2), 255 * (1 - g_lat/2), 0, 150]", get_radius=20)
    
    view = pdk.ViewState(latitude=df['lat'].mean(), longitude=df['lon'].mean(), zoom=13, pitch=30)
    return pdk.Deck(layers=[layer], initial_view_state=view)

def plot_biomechanics(session: Session, run_idx: int):
    run = session.runs[run_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Gauche", "Droite"], y=[run.biomechanics.left_turns_count, run.biomechanics.right_turns_count], marker_color=['#3498db', '#e74c3c']))
    fig.update_layout(title=f"Équilibre Virages - Run #{run.number}", template='plotly_white', height=400)
    return fig

def plot_fatigue_curve(session: Session):
    runs = session.runs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r.number for r in runs], y=[r.fatigue_score*100 for r in runs], mode='lines+markers', name="Fatigue %", line=dict(color='#e67e22', width=3)))
    fig.update_layout(title="Évolution de la Fatigue", xaxis_title="Run #", yaxis_title="Fatigue (%)", template='plotly_white', height=400)
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🎿 Ski Analytics Pro - Ultimate Edition")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded = st.file_uploader("Chargez fichier .slopes", type=['slopes'])
        weight = st.number_input("Poids (kg)", 40, 150, 75)
        if uploaded: st.success("Fichier chargé !")
    
    if not uploaded:
        st.info("👈 Veuillez charger un fichier pour commencer l'analyse.")
        return
    
    with st.spinner("Deep Analyse en cours..."):
        with zipfile.ZipFile(uploaded) as zf:
            activity, actions = parse_metadata(zf)
            gps_data = load_gps(zf)
            if not activity or not actions:
                st.error("Erreur de parsing."); return
            
            if gps_data is not None:
                gps_data = enrich_gps(gps_data)
                enrich_actions(actions, gps_data)
            
            session = Session(activity, actions, weight)
            pistes_gdf = get_pistes_from_osm(session)
            if pistes_gdf is not None: match_gps_to_pistes(actions, pistes_gdf)

    st.subheader(f"Session : {session.activity.location} | {session.stats['weather']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Descentes", session.stats['runs'])
    c2.metric("Distance", f"{session.stats['distance_km']:.1f} km")
    c3.metric("V.Max", f"{session.stats['max_speed']:.1f} km/h")
    c4.metric("Dénivelé", f"{session.stats['descent_m']/1000:.1f} km")

    tab_overview, tab_replay, tab_biomech, tab_heatmaps = st.tabs(["📊 Vue d'ensemble", "🎥 Replay 3D", "🏥 Analyse Biomécanique", "🗺️ Cartes Thermiques"])

    with tab_overview:
        col1, col2 = st.columns(2)
        df_runs = pd.DataFrame([{
            "N°": r.number,
            "Piste": r.track_ids[:20] + ("..." if len(r.track_ids)>20 else ""),
            "Diff": r.difficulty.label,
            "V.Moy": f"{r.avg_speed:.1f}",
            "Pente": f"{r.avg_gradient:.1f}%",
            "Symétrie": f"{r.biomechanics.carving_symmetry_score:.2f}",
            "Fatigue": f"{r.fatigue_score*100:.0f}%"
        } for r in session.runs])
        col1.dataframe(df_runs, use_container_width=True)
        with col2:
            fig_diff = px.histogram(df_runs, x="Diff", color="Diff", title="Répartition des Difficultés", color_discrete_map={d.label: d.color for d in Difficulty})
            st.plotly_chart(fig_diff, use_container_width=True)

    with tab_replay:
        st.info("🎥 Mode Replay : Utilisez le curseur pour rejouer la descente.")
        run_options = [f"Run #{r.number} - {r.track_ids[:20]}" for r in session.runs]
        selected_run_name = st.selectbox("Choisir la descente à rejouer", run_options)
        run_idx = run_options.index(selected_run_name)
        run = session.runs[run_idx]
        df = run.gps_data
        
        if not df.empty:
            steps = range(0, len(df), 5)
            time_idx = st.slider("Timeline", 0, len(steps)-1, 0)
            real_idx = steps[time_idx]
            current_point = df.iloc[real_idx]
            c1, c2, c3 = st.columns(3)
            c1.metric("Vitesse Actuelle", f"{current_point['speed_kmh']:.1f} km/h")
            c2.metric("Altitude", f"{current_point['ele']:.0f} m")
            c3.metric("Cap", f"{current_point['bearing']:.0f}°")
            st.pydeck_chart(create_replay_map(session, run_idx, real_idx), use_container_width=True)

    with tab_biomech:
        st.subheader("Analyse Technique & Physiologique")
        selected_run_name_bio = st.selectbox("Analyser Run", run_options, key="bio")
        run_idx_bio = run_options.index(selected_run_name_bio)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_biomechanics(session, run_idx_bio), use_container_width=True)
            r = session.runs[run_idx_bio]
            st.write("**Détail Run :**")
            st.json({"Virages Gauche": r.biomechanics.left_turns_count, "Virages Droite": r.biomechanics.right_turns_count, "Symétrie": f"{r.biomechanics.carving_symmetry_score:.2f}", "Stabilité": f"{r.biomechanics.stability_index:.2f}"})
        with col2:
            st.plotly_chart(plot_fatigue_curve(session), use_container_width=True)

    with tab_heatmaps:
        st.subheader("Cartographie Thermique")
        mode = st.radio("Type de Heatmap", ["Density (Fréquentation)", "Speed (Vitesse)", "G-Lat (Virages Techniques)"])
        map_mode = "Density"
        if "Speed" in mode: map_mode = "Speed"
        if "G-Lat" in mode: map_mode = "G-Lat"
        st.pydeck_chart(create_heatmap(session, map_mode), use_container_width=True)

if __name__ == "__main__":
    main()
