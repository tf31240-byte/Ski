import streamlit as st
import gpxpy
import gpxpy.gpx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from scipy.signal import savgol_filter
from haversine import haversine
from datetime import datetime, timedelta
import io
import json
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Ski Analytics Pro - Ultimate Edition", 
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
    .big-font { font-size:20px !important; }
    .metric-box { text-align: center; border: 1px solid #ddd; padding: 10px; border-radius: 10px; background-color: #f9f9f9; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
DIFFICULTY_THRESHOLDS = {
    'Verte': {'min': 0, 'max': 12, 'color': '#009E60', 'hex': '#90EE90'},
    'Bleue': {'min': 12, 'max': 25, 'color': '#007FFF', 'hex': '#ADD8E6'},
    'Rouge': {'min': 25, 'max': 45, 'color': '#FF0000', 'hex': '#FFcccb'},
    'Noire': {'min': 45, 'max': 100, 'color': '#000000', 'hex': '#D3D3D3'}
}

# --- UTILITIES & OPTIMIZATION ---

def calculate_distance_vectorized(df):
    """Vectorized Haversine calculation using NumPy (Fast)."""
    lat1, lon1 = np.radians(df['lat']), np.radians(df['lon'])
    lat2, lon2 = np.radians(df['lat'].shift(-1)), np.radians(df['lon'].shift(-1))
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return pd.Series(np.append(c * 6371000, 0)) # Retour en mètres

def smooth_series(series, window_length=21, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth GPS noise.
    Better than rolling mean because it preserves peaks (jumps) better.
    """
    if len(series) < window_length:
        return series
    return savgol_filter(series, window_length=window_length, polyorder=polyorder)

@st.cache_data(ttl=3600)
def get_weather_cached(lat, lon):
    """Cached weather request."""
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

# --- CORE LOGIC (OOP) ---

class SkiRun:
    def __init__(self, run_id, df_segment):
        self.id = run_id
        self.df = df_segment.copy()
        self.start_time = self.df['time'].iloc[0]
        self.end_time = self.df['time'].iloc[-1]
        self._analyze()
        
    def _analyze(self):
        # Calculs basés sur le dataframe déjà enrichi
        self.duration_sec = (self.end_time - self.start_time).total_seconds()
        self.distance_m = self.df['dist'].sum()
        self.drop_m = self.df['ele'].max() - self.df['ele'].min()
        self.max_speed = self.df['speed_kmh'].max()
        self.avg_speed = self.df['speed_kmh'].mean()
        
        # Pente
        self.avg_grade = abs(self.df['gradient'].mean())
        self.max_grade = abs(self.df['gradient'].max())
        
        # Couleur dominante
        color_counts = self.df['color_name'].value_counts()
        self.color = color_counts.index[0] if not color_counts.empty else "Inconnue"
        
        # Forces G
        self.max_g = self.df['g_force_total'].max() if 'g_force_total' in self.df.columns else 0
        
        # Carving Analysis (Rayon de courbure moyen)
        # On estime le carving par la fréquence des virages serrés
        turns = self.df[self.df['turn_angle'].abs() > 30] # Virages > 30 degrés
        if not turns.empty:
            self.avg_turn_angle = turns['turn_angle'].abs().mean()
        else:
            self.avg_turn_angle = 0

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
    def __init__(self, df_raw, user_weight=75):
        self.df_raw = df_raw
        self.user_weight = user_weight
        self.runs = []
        self._process_session()
        
    def _process_session(self):
        df = self.df_raw.copy().reset_index(drop=True)
        
        # 1. Lissage de l'élévation (Physique)
        # Le GPS est bruité, on lisse pour avoir une altitude propre
        df['ele_smooth'] = smooth_series(df['ele'], window_length=21, polyorder=3)
        
        # 2. Calcul des distances et vitesses (Vectorisé)
        df['dist'] = calculate_distance_vectorized(df)
        df['dt'] = df['time'].diff().dt.total_seconds().fillna(0)
        
        # Vitesse = distance / temps * 3.6
        # Protection contre divisions par 0 et sauts temporels GPS
        mask = (df['dt'] > 0) & (df['dt'] < 10) 
        df['speed_raw'] = np.where(mask, (df['dist'] / df['dt']) * 3.6, 0)
        
        # Lissage de la vitesse pour éviter les pics aberrants
        df['speed_kmh'] = smooth_series(df['speed_raw'], window_length=11, polyorder=2)
        df['speed_kmh'] = df['speed_kmh'].clip(0, 150) # Max speed clamp
        
        # 3. Gradient (Pente)
        df['grade_raw'] = np.where(df['dist'] > 0.5, -(df['ele_smooth'].diff() / df['dist']) * 100, 0)
        df['gradient'] = smooth_series(df['grade_raw'], window_length=21, polyorder=3)
        
        # 4. Classification de difficulté
        def get_diff(g):
            g = abs(g)
            for k, v in DIFFICULTY_THRESHOLDS.items():
                if v['min'] <= g < v['max']: return k, v['hex']
            return 'Noire', '#D3D3D3'
            
        diff_data = df['gradient'].apply(get_diff)
        df['color_name'] = [x[0] for x in diff_data]
        df['hex_color'] = [x[1] for x in diff_data]
        
        # 5. Détection d'état (Arret / Remontée / Ski)
        # Amélioré : On regarde la pente et la vitesse
        df['state'] = 'Arret'
        
        # Remontée: vitesse faible, pente montante (ou vitesse constante sur ligne droite télésiège)
        is_lift = (df['speed_kmh'] < 25) & (df['speed_kmh'] > 2) & (df['grade_raw'] > 2)
        df.loc[is_lift, 'state'] = 'Remontee'
        
        # Ski: vitesse suffisante + pente descendante ou mouvement
        is_ski = (df['speed_kmh'] > 5) & (df['grade_raw'] < -1)
        df.loc[is_ski, 'state'] = 'Ski'
        
        # 6. Calcul de l'accélération (G-Force)
        df['accel'] = df['speed_kmh'].diff() / df['dt'].replace(0, np.nan)
        df['g_force'] = (df['accel'] / 9.81).abs() # G latéral (freinage/accélération)
        df['g_force'] = df['g_force'].fillna(0)
        
        # 7. Calcul de l'angle de virage (Carving)
        df['bearing'] = np.arctan2(np.radians(df['lon']).diff(), np.radians(df['lat']).diff()) * 180 / np.pi
        df['turn_angle'] = df['bearing'].diff()
        df['turn_angle'] = df['turn_angle'].clip(-180, 180) # Normalisation

        self.df = df
        self._detect_runs()

    def _detect_runs(self):
        """Segmentation intelligente des descentes."""
        df = self.df
        df['segment'] = (df['state'] != df['state'].shift()).cumsum()
        
        run_id = 1
        for seg_id, group in df.groupby('segment'):
            if group['state'].iloc[0] != 'Ski':
                continue
            
            # Filtres de qualité
            duration = group['dt'].sum()
            drop = group['ele_smooth'].max() - group['ele_smooth'].min()
            distance = group['dist'].sum()
            
            # Ignorer les bouts de pistes trop courts ou avec peu de dénivelé
            if duration < 30 or drop < 20:
                continue
                
            run = SkiRun(run_id, group)
            self.runs.append(run)
            run_id += 1

    def get_global_stats(self):
        total_dist = self.df['dist'].sum() / 1000
        total_descent = self.df[self.df['state']=='Ski']['ele_smooth'].max() - self.df[self.df['state']=='Ski']['ele_smooth'].min()
        
        # Calories approx (MET * poids * temps)
        ski_time_hours = self.df[self.df['state']=='Ski']['dt'].sum() / 3600
        met = 6.5 # Estimation ski alpin
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
            fig.add_trace(go.Scatter(
                x=df_ski[mask]['time'], y=df_ski[mask]['ele_smooth'],
                mode='lines', name=color, line=dict(color=df_ski[mask]['hex_color'].iloc[0], width=2),
                stackgroup='one' # Optionnel pour effet de remplissage, ou None pour lignes simples
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
        # Préparer les données pour la carte
        path_data = []
        
        # Points pour le heatmap (vitesse)
        heatmap_df = df[df['state']=='Ski'][['lon', 'lat', 'speed_kmh']]
        
        # Path pour chaque descente
        for run in runs:
            # On échantillonne les points pour alléger la carte (tous les 5 points)
            path_df = run.df[['lon', 'lat']].iloc[::5]
            path_data.append({"path": path_df.values.tolist(), "color": [255, 255, 255, 200]})

        # Couche de tracé
        path_layer = pdk.Layer(
            "PathLayer",
            data=pd.DataFrame(path_data),
            pickable=True,
            get_path="path",
            get_color="color",
            width_min_pixels=3,
        )

        # Couleur de chaleur
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=heatmap_df,
            get_position=['lon', 'lat'],
            get_weight='speed_kmh',
            radius_pixels=25,
        )

        view_state = pdk.ViewState(
            latitude=df['lat'].mean(),
            longitude=df['lon'].mean(),
            zoom=13,
            pitch=45,
        )

        # Style de carte : Mapbox si clé, sinon Carto (Gratuit)
        if mapbox_token:
            map_style = "mapbox://styles/mapbox/outdoors-v11"
            provider = "mapbox"
        else:
            map_style = pdk.map_styles.CARTO_LIGHT # Style gratuit
            provider = "carto"

        return pdk.Deck(
            layers=[heatmap_layer, path_layer],
            initial_view_state=view_state,
            map_style=map_style,
            map_provider=provider,
            tooltip={"text": "Alt: {ele_smooth}m\nVit: {speed_kmh} km/h"},
            api_key=mapbox_token
        )

# --- MAIN APP ---

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        uploaded_file = st.file_uploader("📂 Charger GPX", type=['gpx'])
        
        st.subheader("Utilisateur")
        user_weight = st.number_input("Poids (kg)", 40, 150, 75)
        
        st.subheader("Préférences")
        mapbox_token = st.text_input("Mapbox Token (Optionnel)", type="password", help="Laisser vide pour utiliser la carte gratuite Carto.")
        
        st.subheader("Filtres")
        show_all_runs = st.checkbox("Voir toutes les descentes sur la carte", value=False)

    if uploaded_file:
        try:
            with st.spinner("Analyse vectorielle et physique en cours..."):
                # Lecture et parsing
                file_bytes = uploaded_file.read()
                # Support encodage binaire ou string
                try:
                     file_str = file_bytes.decode('utf-8')
                except:
                     file_str = file_bytes.decode('latin-1')
                
                gpx = gpxpy.parse(io.StringIO(file_str))
                points = []
                for track in gpx.tracks:
                    for seg in track.segments:
                        for p in seg.segments[0].points: # Simplification structure
                             points.append({'time': p.time, 'lat': p.latitude, 'lon': p.longitude, 'ele': p.elevation})
                
                if not points:
                    st.error("Fichier vide ou format inconnu.")
                    return

                raw_df = pd.DataFrame(points).dropna(subset=['lat', 'lon'])
                
                # Création de la session (Calculs lourds)
                session = SkiSession(raw_df, user_weight)
                
            # Success message
            st.success(f"Session analysée : {len(session.runs)} descentes détectées.")

            # --- ONGLETS ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎿 Liste des Descentes", "📉 Analyse Technique", "🗺️ Carte 3D"])

            with tab1:
                st.header("Vue d'ensemble")
                stats = session.get_global_stats()
                
                # Metrics Grid
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Distance Totale", f"{stats['distance']:.1f} km")
                c2.metric("Dénivelé Positif", f"{stats['descent']:.0f} m")
                c3.metric("Vitesse Max", f"{stats['max_speed']:.1f} km/h")
                c4.metric("Calories", f"{stats['calories']} kcal")
                
                # Météo (Optionnel)
                meteo = get_weather_cached(raw_df['lat'].mean(), raw_df['lon'].mean())
                if meteo:
                    st.info(f"Conditions actuelles estimées : 🌡️ {meteo['temp']}°C | 💨 {meteo['wind']} km/h")

                st.subheader("Graphiques")
                col_plot1, col_plot2 = st.columns(2)
                col_plot1.plotly_chart(Visualizer.plot_elevation(session.df), use_container_width=True)
                col_plot2.plotly_chart(Visualizer.plot_speed(session.df), use_container_width=True)

            with tab2:
                st.header("Détail des descentes")
                
                # Conversion des objets Run en DataFrame pour l'affichage
                run_data = []
                for run in session.runs:
                    m = run.get_metrics()
                    run_data.append({
                        "N°": run.id,
                        "Heure": run.start_time.strftime('%H:%M'),
                        "Couleur": run.color,
                        "Durée": m["Durée"],
                        "Dist (m)": int(run.distance_m),
                        "VMax (km/h)": int(run.max_speed),
                        "G Max": m["G Max"],
                        "Carving (°)": m["Carving (Angle Moy)"]
                    })
                
                df_runs = pd.DataFrame(run_data)
                st.dataframe(df_runs, use_container_width=True)
                
                # Zoom sur une descente
                st.subheader("Analyse détaillée")
                selected_run_id = st.selectbox("Choisir une descente", df_runs['N°'])
                
                run_obj = next(r for r in session.runs if r.id == selected_run_id)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Couleur :** {run_obj.color}\n**Carving :** Virages serrés détectés.")
                with col2:
                    st.metric("Vitesse Max", f"{run_obj.max_speed:.1f} km/h")
                    st.metric("G Max", f"{run_obj.max_g:.2f} G")
                
                # Graphique détaillé de la vitesse
                fig_detail = px.line(run_obj.df, x='time', y='speed_kmh', 
                                      title=f"Vitesse Descente #{run_obj.id}",
                                      labels={'speed_kmh': 'Vitesse (km/h)'})
                st.plotly_chart(fig_detail, use_container_width=True)

            with tab3:
                st.header("Physique & Carving")
                
                st.write("Analyse des forces d'accélération latérales et virages.")
                
                col_g, col_turn = st.columns(2)
                
                # Analyse des G-Forces sur toute la session
                g_data = session.df[session.df['state']=='Ski']
                fig_g = px.histogram(g_data, x='g_force', nbins=50, title="Distribution des Forces G")
                fig_g.update_layout(bargap=0.1)
                col_g.plotly_chart(fig_g, use_container_width=True)
                
                # Analyse Carving
                fig_turn = px.histogram(g_data, x='turn_angle', nbins=60, title="Distribution des Virages (Angles)")
                col_turn.plotly_chart(fig_turn, use_container_width=True)
                
                # Détecteur de sauts (Basé sur l'accélération verticale dérivée - simulé ici par bruit GPS sur alt)
                # Note: Le GPS seul est mauvais pour détecter les sauts. On utilise une heuristique de variation d'altitude rapide.
                df_ski = session.df[session.df['state']=='Ski']
                # Dérivée seconde de l'altitude = accélération verticale approximative
                df_ski['vert_accel'] = df_ski['ele_smooth'].diff().diff() 
                jumps = df_ski[df_ski['vert_accel'] < -0.5] # Seuil arbitraire basé sur le lissage
                
                if not jumps.empty:
                    st.warning(f"{len(jumps)} pics de variation d'altitude détectés (potentiels sauts/bosses).")
                    st.write("*Attention : La détection de saut par GPS GPS est approximative. Le bruit du signal peut créer des faux positifs.*")

            with tab4:
                st.header("Carte Interactive 3D")
                st.caption("Utilisez la souris (Drag) pour tourner, Scroll pour zoomer.")
                
                # Filtrage pour la carte (toutes les descentes ou seulement la sélectionnée)
                runs_to_map = session.runs if show_all_runs else []
                # Note: La fonction create_deck_map utilise session.df pour le heatmap global
                
                deck_chart = Visualizer.create_deck_map(session.df, session.runs, mapbox_token)
                st.pydeck_chart(deck_chart)

        except Exception as e:
            st.error("Erreur lors de l'analyse.")
            st.exception(e)
    else:
        st.title("Ski Analytics Pro")
        st.info("Veuillez charger un fichier GPX (format standard ou export Slopes) dans la barre latérale.")

if __name__ == "__main__":
    main()
