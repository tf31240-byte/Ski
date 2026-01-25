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
    page_title="Ski Analytics Pro - Metadata Final Edition", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLES CSS ---
st.markdown("""
<style>
    .main { padding-top: 2rem; }
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
    lat1, lon1 = np.radians(df['lat'].values), np.radians(df['lon'].values)
    lat2, lon2 = np.radians(df['lat'].shift(-1).values), np.radians(df['lon'].shift(-1).values)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.clip(a, 0, 1)) 
    
    return pd.Series(c * 6371000, index=df.index)

def smooth_series(series, window_length=21, polyorder=2):
    if len(series) < window_length:
        return series
    if window_length % 2 == 0:
        window_length += 1
    if series.isnull().any():
        series = series.interpolate(method='linear').ffill().bfill()
    try:
        result = savgol_filter(series, window_length=window_length, polyorder=polyorder)
        return pd.Series(result, index=series.index)
    except Exception:
        return series

def get_speed_color(speed_kmh):
    speed = np.clip(speed_kmh, 10, 60)
    ratio = (speed - 10) / 50
    r = int(255 * ratio)
    g = int(255 * (1 - ratio))
    b = 0
    return [r, g, b, 200]

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

# --- PARSERS (ROBUSTE) ---

def parse_slope_metadata(zip_file):
    """Extrait les actions Lift/Run du XML."""
    try:
        st.info("📂 1/3 : Extraction des métadonnées (XML)...")
        metadata_files = [f for f in zip_file.namelist() if 'metadata' in f.lower()]
        
        if not metadata_files:
            st.warning("Fichier XML non trouvé dans le ZIP.")
            return None
            
        metadata_file = metadata_files[0]
        with zip_file.open(metadata_file) as f:
            tree = ET.parse(f)
            root = tree.getroot()
            
            segments = []
            for action in root.findall('.//Action'):
                action_type = action.get('type') 
                start_str = action.get('start')
                end_str = action.get('end')
                max_alt = action.get('maxAlt')
                min_alt = action.get('minAlt')
                distance = action.get('distance')
                duration = action.get('duration')
                top_speed = action.get('topSpeed')
                
                if start_str and end_str:
                    try:
                        start_dt = date_parser.parse(start_str)
                        end_dt = date_parser.parse(end_str)
                        start_utc = start_dt.astimezone(timezone.utc)
                        end_utc = end_dt.astimezone(timezone.utc)
                        
                        segments.append({
                            'type': action_type,
                            'start': start_utc,
                            'end': end_utc,
                            'id': action.get('numberOfType'),
                            'max_alt': float(max_alt) if max_alt else None,
                            'min_alt': float(min_alt) if min_alt else None,
                            'distance': float(distance) if distance else None,
                            'duration': float(duration) if duration else None,
                            'top_speed': float(top_speed) if top_speed else None
                        })
                    except Exception:
                        continue
            if segments:
                st.success(f"✅ {len(segments)} segments (Lifts/Runs) détectés.")
            else:
                st.warning("⚠️ Aucun segment 'Action' trouvé dans le XML.")
            return segments
    except Exception as e:
        st.error(f"Erreur parsing XML: {e}")
        return None

def identify_gps_columns(df_raw):
    """Détecte intelligemment les colonnes Time, Lat, Lon, Alt."""
    st.text("🔍 Analyse de la structure du CSV en cours...")
    
    if df_raw.empty:
        st.error("Le fichier CSV est vide.")
        return None

    sample_row = None
    for i in range(min(10, len(df_raw))):
        row_vals = df_raw.iloc[i].values
        valid_floats = 0
        for val in row_vals:
            try:
                float(val)
                valid_floats += 1
            except:
                pass
        if valid_floats >= 3:
            sample_row = row_vals
            break
            
    if sample_row is None:
        st.error("Impossible de trouver une ligne de données valides.")
        st.dataframe(df_raw.head(10))
        return None

    mapping = {'time': None, 'lat': None, 'lon': None, 'ele': None}
    
    for idx, val in enumerate(sample_row):
        try:
            num_val = float(val)
        except:
            continue 
            
        if mapping['time'] is None:
            if num_val > 1_000_000_000: 
                mapping['time'] = idx
                
        if mapping['lat'] is None:
            if 40 <= num_val <= 50: 
                mapping['lat'] = idx
        
        if mapping['lon'] is None:
            if -20 <= num_val <= 10: 
                mapping['lon'] = idx
        
        if mapping['ele'] is None:
            if 0 < num_val < 9000:
                mapping['ele'] = idx

    if all(v is not None for v in mapping.values()):
        st.text(f"Colonnes identifiées -> Time:{mapping['time']}, Lat:{mapping['lat']}, Lon:{mapping['lon']}, Alt:{mapping['ele']}")
        return [mapping['time'], mapping['lat'], mapping['lon'], mapping['ele']]
    else:
        st.error("Échec de l'identification automatique.")
        return None

def load_slope_file(uploaded_file):
    """Charge le fichier .slopes (ZIP)."""
    try:
        st.info("📂 0/3 : Ouverture de l'archive ZIP...")
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            
            # --- 1. XML ---
            metadata_actions = parse_slope_metadata(z)
            if metadata_actions is None:
                pass 

            # --- 2. CSV ---
            st.info("📂 2/3 : Recherche du fichier GPS (CSV)...")
            target_csv = None
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

            st.text(f"Fichier trouvé : {target_csv}")

            with z.open(target_csv) as f:
                df_raw = pd.read_csv(f, sep=r'[|,\t]', engine='python', header=None, on_bad_lines='skip')
                
                if df_raw.empty:
                    st.error("Le fichier CSV est vide.")
                    return None, None

                df_raw.columns = [str(i) for i in range(len(df_raw.columns))]
                
                indices = identify_gps_columns(df_raw)
                
                if indices is None:
                    return None, None
                
                idx_time, idx_lat, idx_lon, idx_ele = indices
                
                max_idx = max(indices)
                if max_idx >= len(df_raw.columns):
                    st.error(f"Erreur critique : Index de colonne {max_idx} supérieur au nombre de colonnes ({len(df_raw.columns)})")
                    return None, None

                df_raw = df_raw.iloc[:, [idx_time, idx_lat, idx_lon, idx_ele]].copy()
                df_raw.columns = ['time', 'lat', 'lon', 'ele']

                # Conversion Types
                df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
                df_raw['lat'] = pd.to_numeric(df_raw['lat'], errors='coerce')
                df_raw['lon'] = pd.to_numeric(df_raw['lon'], errors='coerce')
                df_raw['ele'] = pd.to_numeric(df_raw['ele'], errors='coerce')
                
                df_raw['time'] = pd.to_datetime(df_raw['time'], unit='s', utc=True, errors='coerce')
                
                df_raw = df_raw.dropna(subset=['time', 'lat', 'lon', 'ele'])
                df_raw = df_raw.sort_values('time').reset_index(drop=True)
                
                st.success(f"✅ 3/3 : {len(df_raw)} points GPS chargés avec succès.")

                return df_raw, metadata_actions

    except Exception as e:
        st.error(f"Erreur critique lors de la lecture du fichier Slope: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None, None

# --- CORE LOGIC ---

class SkiRun:
    def __init__(self, run_id, df_segment, metadata_data=None):
        self.id = run_id
        self.df = df_segment.copy()
        self.metadata = metadata_data
        
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
            
        # On utilise les métadonnées XML si disponibles
        if self.metadata:
            self.duration_sec = self.metadata['duration'] if self.metadata['duration'] else (self.end_time - self.start_time).total_seconds()
            self.distance_m = self.metadata['distance'] if self.metadata['distance'] else self.df['dist'].sum()
            self.drop_m = (self.metadata['max_alt'] - self.metadata['min_alt']) if self.metadata.get('max_alt') and self.metadata.get('min_alt') else (self.df['ele'].max() - self.df['ele'].min())
            self.max_speed = self.metadata['top_speed'] if self.metadata['top_speed'] else self.df['speed_kmh'].max()
            self.avg_speed = self.df['speed_kmh'].mean()
        else:
            self.duration_sec = (self.end_time - self.start_time).total_seconds()
            self.distance_m = self.df['dist'].sum()
            self.drop_m = self.df['ele'].max() - self.df['ele'].min()
            self.max_speed = self.df['speed_kmh'].max()
            self.avg_speed = self.df['speed_kmh'].mean()

        # Métriques avancées
        self.max_g_lat = self.df['g_lat'].max()
        self.max_g_long = self.df['g_long'].max()
        self.avg_rugosity = self.df['roughness'].mean()
        self.max_rugosity = self.df['roughness'].max()

        self.avg_grade = abs(self.df['gradient'].mean())
        g = self.avg_grade
        self.color = "Inconnue"
        for k, v in DIFFICULTY_THRESHOLDS.items():
            if v['min'] <= g < v['max']:
                self.color = k
                break
        
        turns = self.df[self.df['turn_angle'].abs() > 30] 
        if not turns.empty:
            self.avg_turn_angle = turns['turn_angle'].abs().mean()
        else:
            self.avg_turn_angle = 0

    def get_metrics(self):
        # Fonction de formatage pour l'affichage
        display_alt = int(self.metadata['max_alt']) if self.metadata and self.metadata['max_alt'] else (int(self.df['ele'].max()) if not self.df.empty else 0)
        display_dist = int(self.metadata['distance']) if self.metadata and self.metadata['distance'] else int(self.distance_m)
        
        return {
            "N°": self.id,
            "Durée": f"{int(self.duration_sec//60)}:{int(self.duration_sec%60):02d}",
            "Distance": f"{display_dist} m",
            "Dénivelé": f"{int(self.drop_m)} m",
            "Altitude (m)": f"{display_alt} m",
            "Vitesse Max": f"{int(self.max_speed)} km/h",
            "Vitesse Moy": f"{self.avg_speed:.1f} km/h",
            "G Lat Max": f"{self.max_g_lat:.2f} G",
            "Rugosité Max": f"{self.max_rugosity:.2f}",
            "Carving": f"{self.avg_turn_angle:.0f}°",
            "Couleur": self.color,
            "Heure": self.start_time.strftime('%H:%M') if self.start_time else "-"
        }

class SkiSession:
    def __init__(self, df_raw, user_weight=75, metadata_actions=None):
        self.df_raw = df_raw
        self.user_weight = user_weight
        self.metadata_actions = metadata_actions
        self.runs = []
        self._process_session()
        
    def _process_session(self):
        df = self.df_raw.copy().reset_index(drop=True)
        
        # 1. Physique de base (indispensable)
        df['ele_smooth'] = smooth_series(df['ele'], window_length=21, polyorder=3)
        df['dist'] = calculate_distance_vectorized(df)
        df['dt'] = df['time'].diff().dt.total_seconds().fillna(0)
        mask = (df['dt'] > 0) & (df['dt'] < 10) 
        df['speed_raw'] = np.where(mask, (df['dist'] / df['dt']) * 3.6, 0)
        df['speed_kmh'] = smooth_series(df['speed_raw'], window_length=11, polyorder=2)
        df['speed_kmh'] = df['speed_kmh'].clip(0, 150)
        df['speed_ms'] = df['speed_kmh'] / 3.6
        
        # 2. Pente & Gradient
        df['grade_raw'] = np.where(df['dist'] > 0.5, -(df['ele_smooth'].diff() / df['dist']) * 100, 0)
        df['gradient'] = smooth_series(df['grade_raw'], window_length=21, polyorder=3)
        
        # 3. Classification difficulté
        def get_diff(g):
            g = abs(g)
            for k, v in DIFFICULTY_THRESHOLDS.items():
                if v['min'] <= g < v['max']: return k, v['hex']
            return 'Noire', '#D3D3D3'
        diff_data = df['gradient'].apply(get_diff)
        df['color_name'] = [x[0] for x in diff_data]
        df['hex_color'] = [x[1] for x in diff_data]

        # 4. FORCE G LONGITUDINALE
        df['accel'] = df['speed_ms'].diff() / df['dt'].replace(0, np.nan)
        df['g_long'] = (df['accel'] / 9.81).abs()
        df['g_long'] = df['g_long'].fillna(0)

        # 5. FORCE G LATÉRALE
        df['bearing'] = np.arctan2(np.radians(df['lon']).diff(), np.radians(df['lat']).diff()) * 180 / np.pi
        df['bearing_diff'] = df['bearing'].diff().fillna(0)
        df['turn_angle_rad'] = np.radians(np.abs(df['bearing_diff']))
        
        mask_turn = df['turn_angle_rad'] > 0.01
        df['radius'] = np.where(mask_turn, df['dist'] / df['turn_angle_rad'], np.inf)
        
        df['centripetal_accel'] = (df['speed_ms']**2) / df['radius']
        df['g_lat'] = (df['centripetal_accel'] / 9.81)
        df['g_lat'] = df['g_lat'].clip(0, 5)

        # 6. RUGOSITÉ
        df['roughness'] = df['ele'].rolling(window=5, center=True).std()
        df['roughness'] = df['roughness'].fillna(0)
        df['turn_angle'] = df['bearing_diff'].clip(-180, 180)

        # 7. Segmentation (XML Truth)
        df['state'] = 'Arret'
        if self.metadata_actions:
            for seg in self.metadata_actions:
                action_type = seg['type'].lower()
                start_t = seg['start']
                end_t = seg['end']
                mask_seg = (df['time'] >= start_t) & (df['time'] <= end_t)
                if mask_seg.any():
                    if action_type == 'lift':
                        df.loc[mask_seg, 'state'] = 'Remontee'
                    elif action_type == 'run':
                        df.loc[mask_seg, 'state'] = 'Ski'
        else:
            # Fallback Heuristique (si pas de métadonnées)
            is_lift = (df['speed_kmh'] < 25) & (df['speed_kmh'] > 2) & (df['grade_raw'] > 0.5)
            df.loc[is_lift, 'state'] = 'Remontee'
            is_ski = (df['speed_kmh'] > 5) & (df['grade_raw'] < -0.5)
            df.loc[is_ski, 'state'] = 'Ski'

        self.df = df
        self._detect_runs()

    def _detect_runs(self):
        df = self.df
        min_drop = st.session_state.get('min_drop', 20.0)
        min_duration = st.session_state.get('min_duration', 30.0)
        
        if df is None: return # Sécurité
        
        df['segment'] = (df['state'] != df['state'].shift()).cumsum()
        run_id = 1
        
        for seg_id, group in df.groupby('segment'):
            if group['state'].iloc[0] != 'Ski':
                continue
            
            duration = group['dt'].sum()
            # Utilisation altitude brute pour le drop
            drop = group['ele'].max() - group['ele'].min() 

            if duration < min_duration or drop < min_drop: continue
            
            # Association métadonnées (trouver l'action correspondante par recouvrement temporel)
            run_meta = None
            if self.metadata_actions:
                for action in self.metadata_actions:
                    if action['type'] == 'Run':
                        start_t = action['start']
                        end_t = action['end']
                        if start_t <= group['time'].min() and end_t >= group['time'].max():
                            run_meta = action
                            break
            
            run = SkiRun(run_id, group, metadata_data=run_meta)
            self.runs.append(run)
            run_id += 1

    def get_global_stats(self):
        if self.df is None: return {}
        
        total_dist = self.df['dist'].sum() / 1000
        df_ski = self.df[self.df['state']=='Ski']
        total_descent = df_ski['ele'].max() - df_ski['ele'].min() if not df_ski.empty else 0
        ski_time_hours = df_ski['dt'].sum() / 3600
        met = 6.5
        calories = int(met * self.user_weight * ski_time_hours)
        
        return {
            "runs": len(self.runs),
            "distance": total_dist,
            "descent": total_descent,
            "max_speed": self.df['speed_kmh'].max(),
            "max_lat_g": df_ski['g_lat'].max() if not df_ski.empty else 0,
            "max_alt": self.df['ele'].max(),
            "calories": calories,
            "duration": f"{int(ski_time_hours)}h {int((ski_time_hours%1)*60)}m"
        }

# --- VISUALIZATION ---
class Visualizer:
    @staticmethod
    def plot_elevation_time(df):
        if df.empty: return go.Figure()
        df_ski = df[df['state']=='Ski']
        if df_ski.empty: return go.Figure()
        
        fig = go.Figure()
        for color in df_ski['color_name'].unique():
            mask = df_ski['color_name'] == color
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df_ski[mask]['time'], y=df_ski[mask]['ele_smooth'],
                    mode='lines', name=color, line=dict(color=df_ski[mask]['hex_color'].iloc[0], width=2),
                    stackgroup='one'
                ))
        fig.update_layout(title="Profil Altitudique (Temps)", yaxis_title="Altitude (m)", xaxis_title="Heure", template='plotly_white', hovermode='x')
        return fig

    @staticmethod
    def plot_elevation_distance(df):
        if df.empty: return go.Figure()
        df['cumul_dist'] = df['dist'].cumsum() / 1000 
        df_ski = df[df['state']=='Ski']
        if df_ski.empty: return go.Figure()
        
        fig = px.line(df_ski, x='cumul_dist', y='ele_smooth', title="Profil Altitudique (Distance)", labels={'cumul_dist': 'Distance (km)', 'ele_smooth': 'Altitude (m)'})
        fig.update_traces(line_color='#007FFF')
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def plot_speed(df):
        if df.empty: return go.Figure()
        df_ski = df[df['state']=='Ski']
        fig = px.line(df_ski, x='time', y='speed_kmh', title="Vitesse", labels={'speed_kmh': 'km/h'})
        fig.update_traces(line_color='#007FFF')
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def plot_g_forces(df):
        if df.empty: return go.Figure()
        df_ski = df[df['state']=='Ski']
        if df_ski.empty: return go.Figure()
        
        df_lat = df_ski[df_ski['g_lat'] > 0.1]['g_lat']
        fig = go.Figure()
        if not df_lat.empty:
            fig.add_trace(go.Histogram(x=df_lat, name='G Latéral (Virage)', opacity=0.75, marker_color='#FF0000'))
        fig.update_layout(title="Distribution des Forces G (Virages)", xaxis_title="G Force", yaxis_title="Fréquence", template='plotly_white', barmode='overlay')
        return fig

    @staticmethod
    def plot_rugosity(df):
        if df.empty: return go.Figure()
        df_ski = df[df['state']=='Ski']
        if df_ski.empty: return go.Figure()
        
        fig = px.line(df_ski, x='time', y='roughness', title="Rugosité (Bosses / Neige)", labels={'roughness': 'Indice de Rugosité'})
        fig.update_traces(line_color='#8B4513')
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def create_deck_map(df, runs, mapbox_token=None):
        if df.empty or not runs: return pdk.Deck()
        path_data = []
        heatmap_df = df[df['state']=='Ski'][['lon', 'lat', 'speed_kmh']].copy()
        if heatmap_df.empty:
            heatmap_df = pd.DataFrame(columns=['lon', 'lat', 'speed_kmh'])
        
        for run in runs:
            path_df = run.df[['lon', 'lat']].iloc[::5]
            avg_speed = run.avg_speed
            color = get_speed_color(avg_speed)
            path_data.append({
                "path": path_df.values.tolist(), 
                "color": color,
                "name": f"Run {run.id} ({avg_speed:.0f} km/h)"
            })

        path_layer = pdk.Layer("PathLayer", data=pd.DataFrame(path_data), pickable=True, get_path="path", get_color="color", width_min_pixels=4, tooltip=True)
        heatmap_layer = pdk.Layer("HeatmapLayer", data=heatmap_df, get_position=['lon', 'lat'], get_weight='speed_kmh', radius_pixels=25)

        view_state = pdk.ViewState(latitude=df['lat'].mean(), longitude=df['lon'].mean(), zoom=13, pitch=45)

        map_style = "mapbox://styles/mapbox/outdoors-v11" if mapbox_token else pdk.map_styles.CARTO_LIGHT
        provider = "mapbox" if mapbox_token else "carto"

        return pdk.Deck(layers=[heatmap_layer, path_layer], initial_view_state=view_state, map_style=map_style, map_provider=provider, tooltip={"text": "{name}\nAlt: {ele_smooth}m\nVit: {speed_kmh} km/h"}, api_keys=mapbox_token)

# --- MAIN APP ---

def main():
    with st.sidebar:
        st.title("⚙️ Configuration")
        uploaded_file = st.file_uploader("📂 Charger fichier .slopes", type=['slopes'])
        
        st.subheader("Utilisateur")
        user_weight = st.number_input("Poids (kg)", 40, 150, 75)
        
        st.subheader("Préférences")
        mapbox_token = st.text_input("Mapbox Token (Optionnel)", type="password")

        st.subheader("Filtres de détection")
        min_drop = st.slider("Dénivelé minimum d'une descente (m)", 0.0, 50.0, 20.0, 1.0)
        # min_duration = st.slider("Durée minimum (sec)", 0.0, 300.0, 30.0, 5.0)
        st.write("Ajustez pour inclure les traversées plates.")
        
        if 'min_drop' not in st.session_state:
            st.session_state['min_drop'] = min_drop

    if uploaded_file:
        try:
            raw_df = None
            metadata = None

            with st.spinner("Analyse des données Slopes..."):
                raw_df, metadata = load_slope_file(uploaded_file)
                if raw_df is None:
                    st.stop()

                if metadata is None:
                    st.warning("⚠️ Les métadonnées XML n'ont pas pu être lues. Utilisation de l'heuristique.")

                session = SkiSession(raw_df, user_weight, metadata_actions=metadata)
                
            st.success(f"Session analysée : {len(session.runs)} descentes détectées.")
            stats = session.get_global_stats()

            # --- REPLAY / SLIDER ---
            st.sidebar.subheader("⏱️ Replay Temporel")
            min_time = session.df['time'].min().to_pydatetime()
            max_time = session.df['time'].max().to_pydatetime()
            total_duration_sec = (max_time - min_time).total_seconds()
            
            time_range = st.sidebar.slider(
                "Sélectionner une plage horaire",
                min_value=0.0,
                max_value=float(total_duration_sec),
                value=(0.0, float(total_duration_sec)),
                step=60.0,
                format="HH:MM"
            )
            
            start_filter = min_time + timedelta(seconds=time_range[0])
            end_filter = min_time + timedelta(seconds=time_range[1])
            
            df_filtered = session.df[(session.df['time'] >= pd.Timestamp(start_filter)) & 
                                     (session.df['time'] <= pd.Timestamp(end_filter))]
            
            if df_filtered.empty:
                st.warning("La sélection temporelle est vide. Affichage de la session complète.")
                df_filtered = session.df
            else:
                st.info(f"Analyse de la période : {start_filter.strftime('%H:%M')} à {end_filter.strftime('%H:%M')}")

            tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎿 Liste des Descentes", "📉 Analyse Technique", "🗺️ Carte 3D"])

            with tab1:
                st.header("Vue d'ensemble")
                stats_f = {
                    "distance": df_filtered['dist'].sum() / 1000,
                    "max_speed": df_filtered['speed_kmh'].max(),
                    "max_lat_g": df_filtered[df_filtered['state']=='Ski']['g_lat'].max() if not df_filtered.empty else 0
                }

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Distance (Sélection)", f"{stats_f['distance']:.2f} km")
                c2.metric("Vitesse Max", f"{stats_f['max_speed']:.1f} km/h")
                c3.metric("G Latéral Max", f"{stats_f['max_lat_g']:.2f} G")
                c4.metric("Calories (Session)", f"{stats['calories']} kcal")
                
                meteo = get_weather_cached(raw_df['lat'].mean(), raw_df['lon'].mean())
                if meteo:
                    st.info(f"Météo estimée : 🌡️ {meteo['temp']}°C | 💨 {meteo['wind']} km/h")

                col_plot1, col_plot2 = st.columns(2)
                col_plot1.plotly_chart(Visualizer.plot_elevation_time(df_filtered), use_container_width=True)
                col_plot2.plotly_chart(Visualizer.plot_elevation_distance(df_filtered), use_container_width=True)

            with tab2:
                st.header("Détail des descentes")
                run_data = []
                for run in session.runs:
                    m = run.get_metrics()
                    run_data.append(m)
                
                df_runs = pd.DataFrame(run_data)
                
                if not df_runs.empty:
                    st.dataframe(df_runs, use_container_width=True)
                    
                    selected_run_id = st.selectbox("Choisir une descente", df_runs['N°'])
                    run_obj = next((r for r in session.runs if r.id == selected_run_id), None)
                    
                    if run_obj and not run_obj.df.empty:
                        col1, col2 = st.columns(2)
                        col1.info(f"**Couleur :** {run_obj.color}\n**Carving :** {run_obj.avg_turn_angle:.0f}° moyen.")
                        col2.metric("Vitesse Max", f"{run_obj.max_speed:.1f} km/h")
                        col2.metric("G Lat Max", f"{run_obj.max_g_lat:.2f} G")
                        fig_detail = px.line(run_obj.df, x='time', y='speed_kmh', title=f"Vitesse Descente #{run_obj.id}")
                        st.plotly_chart(fig_detail, use_container_width=True)
                else:
                    st.warning("Aucune donnée de descente à afficher.")

            with tab3:
                st.header("Physique & Technique")
                df_ski_f = df_filtered[df_filtered['state']=='Ski']
                
                col_g, col_rug = st.columns(2)
                col_g.plotly_chart(Visualizer.plot_g_forces(df_filtered), use_container_width=True)
                col_rug.plotly_chart(Visualizer.plot_rugosity(df_filtered), use_container_width=True)

            with tab4:
                st.header("Carte Interactive 3D")
                deck_chart = Visualizer.create_deck_map(df_filtered, session.runs, mapbox_token)
                st.pydeck_chart(deck_chart)

        except Exception as e:
            st.error("Une erreur critique est survenue.")
            st.exception(e)
    else:
        st.title("Ski Analytics Pro - Metadata Edition")
        st.info("Veuillez charger un fichier .slopes.")

if __name__ == "__main__":
    main()
