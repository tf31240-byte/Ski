"""
SKI ANALYTICS PRO - ARCHITECTURE OPTIMISÉE
==========================================
Version restructurée privilégiant les métadonnées XML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
from scipy.signal import savgol_filter
from datetime import datetime, timezone
import zipfile
import xml.etree.ElementTree as ET
from dateutil import parser as date_parser
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

st.set_page_config(page_title="Ski Analytics Pro", layout="wide")

# ============================================================================
# MODELS
# ============================================================================

class ActionType(Enum):
    RUN = "Run"
    LIFT = "Lift"

class Difficulty(Enum):
    VERTE = ("Verte", 0, 12, "#90EE90")
    BLEUE = ("Bleue", 12, 25, "#ADD8E6")
    ROUGE = ("Rouge", 25, 45, "#FFcccb")
    NOIRE = ("Noire", 45, 100, "#D3D3D3")
    
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
class Activity:
    location: str
    start: datetime
    end: datetime
    distance: float  # m
    duration: int    # s
    vertical: float  # m
    run_count: int
    top_speed: float
    conditions: str

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
    gps_data: pd.DataFrame = None
    difficulty: Difficulty = None
    avg_gradient: float = 0.0
    max_g_lat: float = 0.0
    avg_turn: float = 0.0

# ============================================================================
# PARSING - MÉTADONNÉES PRIORITAIRES
# ============================================================================

def parse_metadata(zip_file: zipfile.ZipFile) -> Tuple[Optional[Activity], List[Action]]:
    """Parse le XML et extrait toutes les métadonnées"""
    try:
        xml_files = [f for f in zip_file.namelist() if 'metadata' in f.lower()]
        if not xml_files:
            st.warning("⚠️ Fichier XML metadata introuvable")
            return None, []
        
        st.info(f"📄 Lecture de {xml_files[0]}")
        
        with zip_file.open(xml_files[0]) as f:
            tree = ET.parse(f)
            root = tree.getroot()
            
            # Affichage debug de la structure
            st.text(f"Root tag: {root.tag}")
            st.text(f"Root attribs: {list(root.attrib.keys())[:5]}")
            
            # Activity globale - le root EST déjà l'Activity
            act_elem = root if root.tag == 'Activity' else root.find('.//Activity')
            
            if act_elem is None:
                st.error("❌ Élément <Activity> introuvable dans le XML")
                st.text("Structure XML:")
                for child in root:
                    st.text(f"  - {child.tag}")
                return None, []
            
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
            
            # Actions (Runs + Lifts)
            actions = []
            actions_elem = root.find('.//actions')
            
            if actions_elem is None:
                st.warning("⚠️ Élément <actions> introuvable, recherche directe des <Action>")
                action_elems = root.findall('.//Action')
            else:
                action_elems = actions_elem.findall('Action')
            
            st.info(f"🔍 {len(action_elems)} actions détectées")
            
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
                        top_speed=float(elem.get('topSpeed', 0))
                    ))
                except Exception as e:
                    st.warning(f"⚠️ Action ignorée: {e}")
                    continue
            
            st.success(f"✅ {len(actions)} actions chargées ({activity.run_count} runs attendus)")
            return activity, actions
            
    except Exception as e:
        st.error(f"❌ Erreur XML: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, []

def load_gps(zip_file: zipfile.ZipFile) -> Optional[pd.DataFrame]:
    """Charge le fichier GPS avec auto-détection"""
    try:
        target = next((f for f in zip_file.namelist() if 'GPS.csv' in f and 'Raw' not in f), None)
        if not target:
            target = next((f for f in zip_file.namelist() if 'RawGPS.csv' in f), None)
        
        if not target:
            return None
        
        with zip_file.open(target) as f:
            df = pd.read_csv(f, sep=r'[|,\t]', engine='python', header=None, on_bad_lines='skip')
            
            # Auto-détection colonnes
            sample = df.iloc[0].values
            indices = {}
            for i, v in enumerate(sample):
                try:
                    num = float(v)
                    if num > 1e9 and 'time' not in indices:
                        indices['time'] = i
                    elif 40 <= num <= 50 and 'lat' not in indices:
                        indices['lat'] = i
                    elif -20 <= num <= 10 and 'lon' not in indices:
                        indices['lon'] = i
                    elif 0 < num < 9000 and 'ele' not in indices:
                        indices['ele'] = i
                except:
                    pass
            
            if len(indices) != 4:
                return None
            
            df = df.iloc[:, [indices['time'], indices['lat'], indices['lon'], indices['ele']]]
            df.columns = ['time', 'lat', 'lon', 'ele']
            
            df['time'] = pd.to_datetime(pd.to_numeric(df['time'], errors='coerce'), unit='s', utc=True)
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df['ele'] = pd.to_numeric(df['ele'], errors='coerce')
            
            df = df.dropna().sort_values('time').reset_index(drop=True)
            st.success(f"✅ {len(df)} points GPS")
            return df
            
    except Exception as e:
        st.error(f"Erreur GPS: {e}")
        return None

# ============================================================================
# ENRICHISSEMENT GPS
# ============================================================================

def enrich_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute calculs physiques avancés"""
    df = df.copy()
    
    # Distance Haversine
    lat1, lon1 = np.radians(df['lat']), np.radians(df['lon'])
    lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    df['dist'] = 2 * np.arcsin(np.sqrt(a.clip(0, 1))) * 6371000
    
    # Vitesse
    df['dt'] = df['time'].diff().dt.total_seconds().fillna(0)
    df['speed_kmh'] = np.where((df['dt'] > 0) & (df['dt'] < 10), 
                               (df['dist'] / df['dt']) * 3.6, 0).clip(0, 150)
    df['speed_ms'] = df['speed_kmh'] / 3.6
    
    # Pente
    df['ele_smooth'] = savgol_filter(df['ele'], 21, 3) if len(df) > 21 else df['ele']
    df['gradient'] = np.where(df['dist'] > 0.5, 
                             -(df['ele_smooth'].diff() / df['dist']) * 100, 0)
    
    # Forces G latérales
    df['bearing'] = np.arctan2(np.radians(df['lon']).diff(), 
                              np.radians(df['lat']).diff()) * 180 / np.pi
    df['turn_rad'] = np.radians(df['bearing'].diff().abs())
    df['radius'] = np.where(df['turn_rad'] > 0.01, df['dist'] / df['turn_rad'], np.inf)
    df['g_lat'] = ((df['speed_ms']**2 / df['radius']) / 9.81).clip(0, 5)
    df['turn_angle'] = df['bearing'].diff().clip(-180, 180)
    
    return df

def enrich_actions(actions: List[Action], gps: pd.DataFrame):
    """Associe GPS aux actions et calcule métriques"""
    for action in actions:
        mask = (gps['time'] >= action.start) & (gps['time'] <= action.end)
        action.gps_data = gps[mask].copy()
        
        if action.type == ActionType.RUN and not action.gps_data.empty:
            df = action.gps_data
            action.avg_gradient = abs(df['gradient'].mean())
            action.difficulty = Difficulty.from_gradient(action.avg_gradient)
            action.max_g_lat = df['g_lat'].max()
            turns = df[df['turn_angle'].abs() > 30]
            action.avg_turn = turns['turn_angle'].abs().mean() if not turns.empty else 0

# ============================================================================
# SESSION
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
        return {
            "location": self.activity.location,
            "runs": len(self.runs),
            "distance_km": self.activity.distance / 1000,
            "descent_m": sum(abs(r.vertical) for r in self.runs),
            "duration": f"{self.activity.duration//3600}h{(self.activity.duration%3600)//60}m",
            "max_speed": max((r.top_speed for r in self.runs), default=0),
            "max_g": max((r.max_g_lat for r in self.runs), default=0),
            "calories": int(6.5 * self.weight * ski_hours)
        }
    
    def runs_df(self):
        data = []
        for r in self.runs:
            data.append({
                "N°": r.number,
                "Heure": r.start.strftime('%H:%M'),
                "Durée": f"{int(r.duration//60)}:{int(r.duration%60):02d}",
                "Distance": f"{int(r.distance)}m",
                "Dénivelé": f"{int(abs(r.vertical))}m",
                "V.Max": f"{r.top_speed:.1f}",
                "V.Moy": f"{r.avg_speed:.1f}",
                "Pente": f"{r.avg_gradient:.1f}%",
                "Couleur": r.difficulty.label if r.difficulty else "-",
                "G Lat": f"{r.max_g_lat:.2f}",
                "Carving": f"{r.avg_turn:.0f}°"
            })
        return pd.DataFrame(data)

# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_elevation(session: Session):
    fig = go.Figure()
    for r in session.runs:
        if r.gps_data is not None and not r.gps_data.empty:
            fig.add_trace(go.Scatter(
                x=r.gps_data['time'], 
                y=r.gps_data['ele_smooth'],
                mode='lines',
                name=f"Run {r.number}",
                line=dict(color=r.difficulty.color if r.difficulty else '#CCC', width=2)
            ))
    fig.update_layout(title="Profil Altimétrique", yaxis_title="Alt (m)", template='plotly_white')
    return fig

def plot_speed(session: Session):
    fig = go.Figure()
    for r in session.runs:
        if r.gps_data is not None and not r.gps_data.empty:
            fig.add_trace(go.Scatter(
                x=r.gps_data['time'], 
                y=r.gps_data['speed_kmh'],
                mode='lines',
                name=f"Run {r.number}"
            ))
    fig.update_layout(title="Vitesse", yaxis_title="km/h", template='plotly_white')
    return fig

def create_map(session: Session):
    paths = []
    for r in session.runs:
        if r.gps_data is not None and not r.gps_data.empty:
            df = r.gps_data[['lon', 'lat']].iloc[::5]
            speed_ratio = min(r.avg_speed / 60, 1)
            color = [int(255*speed_ratio), int(255*(1-speed_ratio)), 50, 200]
            paths.append({
                "path": df.values.tolist(),
                "color": color,
                "name": f"Run {r.number}"
            })
    
    layer = pdk.Layer("PathLayer", pd.DataFrame(paths), get_path="path", 
                     get_color="color", width_min_pixels=4)
    
    all_gps = pd.concat([r.gps_data for r in session.runs if r.gps_data is not None])
    view = pdk.ViewState(latitude=all_gps['lat'].mean(), 
                        longitude=all_gps['lon'].mean(), zoom=13, pitch=45)
    
    return pdk.Deck(layers=[layer], initial_view_state=view)

# ============================================================================
# APP PRINCIPALE
# ============================================================================

def main():
    st.title("⛷️ Ski Analytics Pro - Metadata Edition")
    
    with st.sidebar:
        st.header("Configuration")
        uploaded = st.file_uploader("Fichier .slopes", type=['slopes'])
        weight = st.number_input("Poids (kg)", 40, 150, 75)
    
    if not uploaded:
        st.info("👈 Chargez un fichier .slopes")
        return
    
    with st.spinner("Analyse..."):
        with zipfile.ZipFile(uploaded) as zf:
            activity, actions = parse_metadata(zf)
            gps_data = load_gps(zf)
            
            if not activity or not actions:
                st.error("Échec parsing métadonnées")
                return
            
            if gps_data is not None:
                gps_data = enrich_gps(gps_data)
                enrich_actions(actions, gps_data)
            
            session = Session(activity, actions, weight)
    
    st.success(f"Session {session.activity.location} analysée")
    
    # Stats globales
    stats = session.stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Descentes", stats['runs'])
    c2.metric("Distance", f"{stats['distance_km']:.1f} km")
    c3.metric("V.Max", f"{stats['max_speed']:.1f} km/h")
    c4.metric("Calories", stats['calories'])
    
    tab1, tab2, tab3 = st.tabs(["📊 Vue d'ensemble", "🎿 Descentes", "🗺️ Carte"])
    
    with tab1:
        col1, col2 = st.columns(2)
        col1.plotly_chart(plot_elevation(session), use_container_width=True)
        col2.plotly_chart(plot_speed(session), use_container_width=True)
    
    with tab2:
        df_runs = session.runs_df()
        st.dataframe(df_runs, use_container_width=True)
        
        if not df_runs.empty:
            selected = st.selectbox("Détail descente", df_runs['N°'])
            run = next(r for r in session.runs if r.number == selected)
            
            if run.gps_data is not None:
                st.plotly_chart(
                    go.Figure(go.Scatter(x=run.gps_data['time'], y=run.gps_data['speed_kmh'])),
                    use_container_width=True
                )
    
    with tab3:
        st.pydeck_chart(create_map(session))

if __name__ == "__main__":
    main()
