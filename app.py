import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from scipy.signal import savgol_filter
from datetime import datetime, timedelta, timezone
import zipfile
import requests
import xml.etree.ElementTree as ET
from dateutil import parser as date_parser

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Ski Analytics Pro - Slope Edition", 
    layout="wide", 
    page_icon="🏔️",
    initial_sidebar_state="expanded"
)

# --- STYLES CSS ---
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
DIFFICULTY_THRESHOLDS = {
    'Verte': {'min': 0, 'max': 12, 'color': '#009E60', 'hex': '#90EE90'},
    'Bleue': {'min': 12, 'max': 25, 'color': '#007FFF', 'hex': '#ADD8E6'},
    'Rouge': {'min': 25, 'max': 45, 'color': '#FF0000', 'hex': '#FFcccb'},
    'Noire': {'min': 45, 'max': 100, 'color': '#000000', 'hex': '#D3D3D3'}
}

# --- UTILITIES ---

def calculate_distance_vectorized(df):
    """Calcul vectorisé de la distance Haversine."""
    lat1, lon1 = np.radians(df['lat']), np.radians(df['lon'])
    lat2, lon2 = np.radians(df['lat'].shift(-1)), np.radians(df['lon'].shift(-1))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return pd.Series(np.append(c * 6371000, 0))

def smooth_series(series, window_length=21, polyorder=2):
    if len(series) < window_length:
        return series
    return savgol_filter(series, window_length=window_length, polyorder=polyorder)

@st.cache_data(ttl=3600)
def get_weather_cached(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                'temp': data['current_weather']['temperature'],
                'wind': data['current_weather']['windspeed']
            }
    except:
        pass
    return None

# --- SLOPE PARSERS (XML & CSV) ---

def parse_slope_metadata(zip_file):
    """
    Extrait et analyse le fichier Metadata (XML) du ZIP.
    Retourne une liste de segments (Lift/Run) avec des timestamps UTC.
    """
    try:
        # Chercher le fichier Metadata
        metadata_files = [f for f in zip_file.namelist() if 'metadata' in f.lower()]
        
        if not metadata_files:
            return None
            
        metadata_file = metadata_files[0]
        
        with zip_file.open(metadata_file) as f:
            tree = ET.parse(f)
            root = tree.getroot()
            
            segments = []
            # Parser chaque Action (Run ou Lift)
            for action in root.findall('.//Action'):
                action_type = action.get('type') # 'Lift' ou 'Run'
                start_str = action.get('start')
                end_str = action.get('end')
                
                if start_str and end_str:
                    try:
                        # Parsing de la date (ex: 2026-01-23 09:14:48 +0100)
                        start_dt = date_parser.parse(start_str)
                        end_dt = date_parser.parse(end_str)
                        
                        # Conversion en UTC pour matcher les timestamps Unix du CSV
                        start_utc = start_dt.astimezone(timezone.utc)
                        end_utc = end_dt.astimezone(timezone.utc)
                        
                        segments.append({
                            'type': action_type,
                            'start': start_utc,
                            'end': end_utc,
                            'id': action.get('numberOfType')
                        })
                    except Exception as e:
                        continue # Ignorer les segments mal formatés
            
            return segments
            
    except Exception as e:
        return None

def load_slope_file(uploaded_file):
    """Charge le fichier .slope (ZIP) et extrait le CSV GPS + le XML Metadata."""
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            # 1. Charger les Métadonnées (XML)
            metadata_segments = parse_slope_metadata(z)
            
            if not metadata_segments:
                st.error("Fichier Slope invalide : Métadonnées XML manquantes ou illisibles.")
                return None, None

            # 2. Charger les données GPS (CSV)
            target_csv = None
            # Préférence pour GPS.csv (plus propre) sinon RawGPS.csv
            for fname in z.namelist():
                if 'GPS.csv' in fname and 'Raw' not in fname:
                    target_csv = fname
                    break
            if not target_csv:
                for fname in z.namelist():
                    if 'RawGPS.csv' in fname:
                        target_csv = fname
                        break

            if not target_csv:
                st.error("Aucun fichier GPS (GPS.csv ou RawGPS.csv) trouvé.")
                return None, None

            with z.open(target_csv) as f:
                # Lecture robuste CSV
                # On lit la première ligne pour détecter la structure
                first_line = f.readline().decode('utf-8').strip()
                f.seek(0)
                
                parts = [p.strip() for p in first_line.split('|')]
                
                # Détection intelligente : Si la 2ème colonne est un timestamp > 2001, pas de header
                has_header = True
                try:
                    val = float(parts[1])
                    if val > 1_000_000_000:
                        has_header = False
                except:
                    pass
                
                if not has_header:
                    # Lecture sans header (ordre standard: Index, Time, Lat, Lon, Alt)
                    df_raw = pd.read_csv(f, sep='|', header=None, engine='python')
                    # On prend les colonnes 1, 2, 3, 4
                    df_raw = df_raw.iloc[:, [1, 2, 3, 4]] 
                    df_raw.columns = ['time', 'lat', 'lon', 'ele']
                else:
                    df_raw = pd.read_csv(f, sep='|', engine='python')
                    df_raw.columns = [c.strip() for c in df_raw.columns]
                    
                    # Mapping des colonnes par nom
                    mapping = {}
                    for c in df_raw.columns:
                        cl = c.lower()
                        if 'time' in cl or 'timestamp' in cl: mapping['time'] = c
                        elif 'lat' in cl: mapping['lat'] = c
                        elif 'lon' in cl or 'long' in cl: mapping['lon'] = c
                        elif 'alt' in cl or 'ele' in cl or 'elevation' in cl: mapping['ele'] = c
                    
                    if len(mapping) == 4:
                        df_raw = df_raw[list(mapping.values())]
                        df_raw.columns = ['time', 'lat', 'lon', 'ele']
                    else:
                        # Fallback positionnel
                        df_raw = df_raw.iloc[:, [1, 2, 3, 4]]
                        df_raw.columns = ['time', 'lat', 'lon', 'ele']

                # Conversion des types
                # Les timestamps Slope sont en secondes Unix
                df_raw['time'] = pd.to_datetime(df_raw['time'], unit='s', utc=True, errors='coerce')
                df_raw['lat'] = pd.to_numeric(df_raw['lat'], errors='coerce')
                df_raw['lon'] = pd.to_numeric(df_raw['lon'], errors='coerce')
                df_raw['ele'] = pd.to_numeric(df_raw['ele'], errors='coerce')
                
                # Nettoyage
                df_raw = df_raw.dropna(subset=['time', 'lat', 'lon', 'ele'])
                df_raw = df_raw.sort_values('time').reset_index(drop=True)

                return df_raw, metadata_segments

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Slope: {e}")
        return None, None

# --- CORE LOGIC (OPTIMIZED FOR XML) ---

class SkiRun:
    def __init__(self, run_id, df_segment):
        self.id = run_id
        self.df = df_segment.copy()
        if not self.df.empty:
            self.start_time = self.df['time'].iloc[0]
            self.end_time = self.df['time'].iloc[-1]
        else:
            self.start_time = None
            self.end_time = None
        self._analyze()
        
    def _analyze(self):
        if self.df.empty:
            return
            
        self.duration_sec = (self.end_time - self.start_time).total_seconds()
        self.distance_m = self.df['dist'].sum()
        self.drop_m = self.df['ele'].max() - self.df['ele'].min()
        self.max_speed = self.df['speed_kmh'].max()
        self.avg_speed = self.df['speed_kmh'].mean()
        self.avg_grade = abs(self.df['gradient'].mean())
        self.max_grade = abs(self.df['gradient'].max())
        
        # Couleur basée sur la pente moyenne
        g = abs(self.avg_grade)
        self.color = "Inconnue"
        for k, v in DIFFICULTY_THRESHOLDS.items():
            if v['min'] <= g < v['max']:
                self.color = k
                break
        
        # Carving (virages > 30°)
        turns = self.df[self.df['turn_angle'].abs() > 30] 
        if not turns.empty:
            self.avg_turn_angle = turns['turn_angle'].abs().mean()
        else:
            self.avg_turn_angle = 0
            
        self.max_g = self.df['g_force_total'].max() if 'g_force_total' in self.df.columns else 0

    def get_metrics(self):
        return {
            "Durée": f"{int(self.duration_sec//60)}:{int(self.duration_sec%60):02d}",
            "Distance": f"{int(self.distance_m)} m",
            "Dénivelé": f"{int(self.drop_m)} m",
            "Vitesse Max": f"{int(self.max_speed)} km/h",
            "Vitesse Moy": f"{self.avg_speed:.1f} km/h",
            "Pente Max": f"{self.max_grade:.1f} %",
            "Carving (Angle Moy)": f"{self.avg_turn_angle:.1f}°",
            "G Max": f"{self.max_g:.2f} G"
        }

class SkiSession:
    def __init__(self, df_raw, user_weight=75, metadata_segments=None):
        self.df_raw = df_raw
        self.user_weight = user_weight
        self.metadata_segments = metadata_segments
        self.runs = []
        self._process_session()
        
    def _process_session(self):
        df = self.df_raw.copy().reset_index(drop=True)
        
        # 1. Calculs Physiques communs
        df['ele_smooth'] = smooth_series(df['ele'], window_length=21, polyorder=3)
        df['dist'] = calculate_distance_vectorized(df)
        df['dt'] = df['time'].diff().dt.total_seconds().fillna(0)
        mask = (df['dt'] > 0) & (df['dt'] < 10) 
        df['speed_raw'] = np.where(mask, (df['dist'] / df['dt']) * 3.6, 0)
        df['speed_kmh'] = smooth_series(df['speed_raw'], window_length=11, polyorder=2)
        df['speed_kmh'] = df['speed_kmh'].clip(0, 150)
        
        df['grade_raw'] = np.where(df['dist'] > 0.5, -(df['ele_smooth'].diff() / df['dist']) * 100, 0)
        df['gradient'] = smooth_series(df['grade_raw'], window_length=21, polyorder=3)
        
        def get_diff(g):
            g = abs(g)
            for k, v in DIFFICULTY_THRESHOLDS.items():
                if v['min'] <= g < v['max']: return k, v['hex']
            return 'Noire', '#D3D3D3'
        diff_data = df['gradient'].apply(get_diff)
        df['color_name'] = [x[0] for x in diff_data]
        df['hex_color'] = [x[1] for x in diff_data]
        
        df['accel'] = df['speed_kmh'].diff() / df['dt'].replace(0, np.nan)
        df['g_force'] = (df['accel'] / 9.81).abs()
        df['g_force'] = df['g_force'].fillna(0)
        df['bearing'] = np.arctan2(np.radians(df['lon']).diff(), np.radians(df['lat']).diff()) * 180 / np.pi
        df['turn_angle'] = df['bearing'].diff()
        df['turn_angle'] = df['turn_angle'].clip(-180, 180)

        # 2. Segmentation basée sur le METADATA XML (La Vérité)
        # Initialisation à 'Arret' par défaut (pour ce qui n'est pas dans le XML)
        df['state'] = 'Arret'
        
        if self.metadata_segments:
            # On applique les segments XML
            for seg in self.metadata_segments:
                action_type = seg['type'].lower() # 'lift' ou 'run'
                start_t = seg['start']
                end_t = seg['end']
                
                # Création du masque temporel
                mask_seg = (df['time'] >= start_t) & (df['time'] <= end_t)
                
                if action_type == 'lift':
                    df.loc[mask_seg, 'state'] = 'Remontee'
                elif action_type == 'run':
                    df.loc[mask_seg, 'state'] = 'Ski'
        else:
            # Cas d'erreur critique : Fichier Slope sans Metadata XML exploitable
            st.error("Les métadonnées XML sont manquantes. La segmentation est impossible.")
            # On ne peut pas continuer l'analyse proprement sans info de segment
            self.df = df
            return

        self.df = df
        self._detect_runs()

    def _detect_runs(self):
        """Crée les objets SkiRun basés sur la colonne 'state'."""
        df = self.df
        df['segment'] = (df['state'] != df['state'].shift()).cumsum()
        
        run_id = 1
        for seg_id, group in df.groupby('segment'):
            if group['state'].iloc[0] != 'Ski':
                continue
            
            duration = group['dt'].sum()
            drop = group['ele_smooth'].max() - group['ele_smooth'].min()
            
            # Filtre qualité (ignore les micro-séquences ski)
            if duration < 30 or drop < 20:
                continue
                
            run = SkiRun(run_id, group)
            self.runs.append(run)
            run_id += 1

    def get_global_stats(self):
        total_dist = self.df['dist'].sum() / 1000
        df_ski = self.df[self.df['state']=='Ski']
        
        total_descent = 0
        if not df_ski.empty:
            total_descent = df_ski['ele_smooth'].max() - df_ski['ele_smooth'].min()

        ski_time_hours = df_ski['dt'].sum() / 3600
        met = 6.5
        calories = int(met * self.user_weight * ski_time_hours)
        
        return {
            "runs": len(self.runs),
            "distance": total_dist,
            "descent": total_descent,
            "max_speed": self.df['speed_kmh'].max(),
            "max_alt": self.df['ele_smooth'].max(),
            "calories": calories,
            "duration": f"{int(ski_time_hours)}h {int((ski_time_hours%1)*60)}m"
        }

# --- VISUALIZATION ---
class Visualizer:
    @staticmethod
    def plot_elevation(df):
        df_ski = df[df['state']=='Ski']
        fig = go.Figure()
        for color in df_ski['color_name'].unique():
            mask = df_ski['color_name'] == color
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df_ski[mask]['time'], y=df_ski[mask]['ele_smooth'],
                    mode='lines', name=color, line=dict(color=df_ski[mask]['hex_color'].iloc[0], width=2),
                    stackgroup='one'
                ))
        fig.update_layout(title="Profil Altitudique", yaxis_title="Altitude (m)", xaxis_title="Heure", template='plotly_white', hovermode='x')
        return fig

    @staticmethod
    def plot_speed(df):
        df_ski = df[df['state']=='Ski']
        fig = px.line(df_ski, x='time', y='speed_kmh', title="Vitesse (Lissée)", labels={'speed_kmh': 'km/h'})
        fig.update_traces(line_color='#007FFF')
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def create_deck_map(df, runs, mapbox_token=None):
        path_data = []
        heatmap_df = df[df['state']=='Ski'][['lon', 'lat', 'speed_kmh']]
        
        for run in runs:
            path_df = run.df[['lon', 'lat']].iloc[::5]
            path_data.append({"path": path_df.values.tolist(), "color": [255, 255, 255, 200]})

        path_layer = pdk.Layer("PathLayer", data=pd.DataFrame(path_data), pickable=True, get_path="path", get_color="color", width_min_pixels=3)
        heatmap_layer = pdk.Layer("HeatmapLayer", data=heatmap_df, get_position=['lon', 'lat'], get_weight='speed_kmh', radius_pixels=25)

        view_state = pdk.ViewState(latitude=df['lat'].mean(), longitude=df['lon'].mean(), zoom=13, pitch=45)
        map_style = "mapbox://styles/mapbox/outdoors-v11" if mapbox_token else pdk.map_styles.CARTO_LIGHT
        provider = "mapbox" if mapbox_token else "carto"

        return pdk.Deck(layers=[heatmap_layer, path_layer], initial_view_state=view_state, map_style=map_style, map_provider=provider, tooltip={"text": "Alt: {ele_smooth}m\nVit: {speed_kmh} km/h"}, api_key=mapbox_token)

# --- MAIN APP ---

def main():
    with st.sidebar:
        st.title("⚙️ Configuration")
        # Uniquement le format .slope maintenant
        uploaded_file = st.file_uploader("📂 Charger fichier Slope", type=['slope'])
        
        st.subheader("Utilisateur")
        user_weight = st.number_input("Poids (kg)", 40, 150, 75)
        
        st.subheader("Préférences")
        mapbox_token = st.text_input("Mapbox Token (Optionnel)", type="password")

    if uploaded_file:
        try:
            raw_df = None
            metadata = None

            with st.spinner("Analyse des données Slope en cours..."):
                # Chargement unifié Slope
                raw_df, metadata = load_slope_file(uploaded_file)
                
                if raw_df is None or metadata is None:
                    st.stop()

                # Création de la session
                session = SkiSession(raw_df, user_weight, metadata_segments=metadata)
                
            st.success(f"Session analysée : {len(session.runs)} descentes détectées (via XML).")

            tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎿 Liste des Descentes", "📉 Analyse Technique", "🗺️ Carte 3D"])

            with tab1:
                st.header("Vue d'ensemble")
                stats = session.get_global_stats()
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Distance Totale", f"{stats['distance']:.1f} km")
                c2.metric("Dénivelé Positif", f"{stats['descent']:.0f} m")
                c3.metric("Vitesse Max", f"{stats['max_speed']:.1f} km/h")
                c4.metric("Calories", f"{stats['calories']} kcal")
                
                meteo = get_weather_cached(raw_df['lat'].mean(), raw_df['lon'].mean())
                if meteo:
                    st.info(f"Météo estimée : 🌡️ {meteo['temp']}°C | 💨 {meteo['wind']} km/h")

                col_plot1, col_plot2 = st.columns(2)
                col_plot1.plotly_chart(Visualizer.plot_elevation(session.df), use_container_width=True)
                col_plot2.plotly_chart(Visualizer.plot_speed(session.df), use_container_width=True)

            with tab2:
                st.header("Détail des descentes")
                run_data = []
                for run in session.runs:
                    m = run.get_metrics()
                    run_data.append({
                        "N°": run.id, 
                        "Heure": run.start_time.strftime('%H:%M') if run.start_time else "-",
                        "Couleur": run.color, 
                        "Durée": m["Durée"], 
                        "Dist (m)": int(run.distance_m),
                        "VMax (km/h)": int(run.max_speed), 
                        "G Max": m["G Max"], 
                        "Carving (°)": m["Carving (Angle Moy)"]
                    })
                df_runs = pd.DataFrame(run_data)
                st.dataframe(df_runs, use_container_width=True)
                
                selected_run_id = st.selectbox("Choisir une descente", df_runs['N°'])
                run_obj = next((r for r in session.runs if r.id == selected_run_id), None)
                
                if run_obj:
                    col1, col2 = st.columns(2)
                    col1.info(f"**Couleur :** {run_obj.color}\n**Carving :** {run_obj.avg_turn_angle:.1f}° moyen.")
                    col2.metric("Vitesse Max", f"{run_obj.max_speed:.1f} km/h")
                    col2.metric("G Max", f"{run_obj.max_g:.2f} G")
                    fig_detail = px.line(run_obj.df, x='time', y='speed_kmh', title=f"Vitesse Descente #{run_obj.id}")
                    st.plotly_chart(fig_detail, use_container_width=True)

            with tab3:
                st.header("Physique & Carving")
                g_data = session.df[session.df['state']=='Ski']
                col_g, col_turn = st.columns(2)
                fig_g = px.histogram(g_data, x='g_force', nbins=50, title="Distribution des Forces G")
                col_g.plotly_chart(fig_g, use_container_width=True)
                fig_turn = px.histogram(g_data, x='turn_angle', nbins=60, title="Distribution des Virages")
                col_turn.plotly_chart(fig_turn, use_container_width=True)

            with tab4:
                st.header("Carte Interactive 3D")
                deck_chart = Visualizer.create_deck_map(session.df, session.runs, mapbox_token)
                st.pydeck_chart(deck_chart)

        except Exception as e:
            st.error("Erreur lors de l'analyse.")
            st.exception(e)
    else:
        st.title("Ski Analytics Pro - Slope Only")
        st.info("Veuillez charger un fichier .slope pour commencer.")

if __name__ == "__main__":
    main()
