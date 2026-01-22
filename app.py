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

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Ski Analytics Pro - AI Edition", 
    layout="wide", 
    page_icon="🏔️",
    initial_sidebar_state="expanded"
)

# --- STYLES CSS AMÉLIORÉS ---
st.markdown("""
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
""", unsafe_allow_html=True)

# --- CONFIGURATION AVANCÉE ---
DIFFICULTY_THRESHOLDS = {
    'verte': {'min': 0, 'max': 12, 'color': '#009E60', 'hex': '#90EE90', 'score': 1},
    'bleue': {'min': 12, 'max': 25, 'color': '#007FFF', 'hex': '#ADD8E6', 'score': 2},
    'rouge': {'min': 25, 'max': 45, 'color': '#FF0000', 'hex': '#FFcccb', 'score': 3},
    'noire': {'min': 45, 'max': 100, 'color': '#000000', 'hex': '#D3D3D3', 'score': 4}
}

# Limites de filtrage
MAX_SPEED_KMH = 140
MIN_RUN_DURATION = 30  # secondes
MIN_RUN_DROP = 30  # mètres
SMOOTHING_WINDOW = 10

# --- FONCTIONS UTILITAIRES AMÉLIORÉES ---

def get_piste_difficulty(gradient):
    """
    Retourne la difficulté de la piste selon le gradient
    Args:
        gradient (float): Pente en pourcentage
    Returns:
        tuple: (nom_couleur, code_hex, score_difficulté)
    """
    gradient = abs(gradient)  # On travaille avec la valeur absolue
    
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
    """
    Lisse une série de données
    Args:
        series: Série pandas à lisser
        window: Taille de la fenêtre
        method: 'rolling' ou 'ewm' (exponential weighted moving average)
    """
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
    """
    Interroge l'API Overpass OSM pour trouver le nom de la piste
    Args:
        lat, lon: Coordonnées GPS
        timeout: Timeout de la requête
    Returns:
        str: Nom de la piste ou message par défaut
    """
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

def parse_and_enrich_gpx(file_content):
    """
    Parse un fichier GPX et enrichit les données avec tous les calculs
    Args:
        file_content: Contenu du fichier GPX (string ou file object)
    Returns:
        DataFrame enrichi avec toutes les métriques
    """
    try:
        # Parse GPX
        if isinstance(file_content, str):
            gpx = gpxpy.parse(file_content)
        else:
            gpx = gpxpy.parse(file_content)
        
        # Extraction des points
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
        
        # Tri par temps
        df = df.sort_values('time').reset_index(drop=True)
        
        # --- CALCULS DE BASE ---
        df['prev_lat'] = df['lat'].shift(1)
        df['prev_lon'] = df['lon'].shift(1)
        df['prev_ele'] = df['ele'].shift(1)
        df['prev_time'] = df['time'].shift(1)
        
        # Distance (avec gestion des erreurs)
        df['dist'] = df.apply(
            lambda x: calculate_distance(x['lat'], x['lon'], x['prev_lat'], x['prev_lon']),
            axis=1
        )
        
        # Temps écoulé
        df['dt'] = (df['time'] - df['prev_time']).dt.total_seconds().fillna(0)
        
        # --- VITESSE ---
        df['speed_raw'] = np.where(
            df['dt'] > 0,
            (df['dist'] / df['dt']) * 3.6,  # Conversion en km/h
            0
        )
        
        # Nettoyage des vitesses aberrantes
        df['speed_kmh'] = detect_outliers_speed(df['speed_raw'])
        
        # Lissage de la vitesse
        df['speed_kmh'] = smooth_data(df['speed_kmh'], window=5, method='ewm')
        
        # --- DÉNIVELÉ ET GRADIENT ---
        df['ele_diff'] = df['ele'] - df['prev_ele']
        
        # Calcul du gradient (pente en %)
        df['gradient'] = np.where(
            df['dist'] > 0.5,  # Ignore les micro-mouvements
            -(df['ele_diff'] / df['dist']) * 100,
            0
        )
        
        # Lissage du gradient
        df['gradient'] = smooth_data(df['gradient'].clip(-100, 100), window=15)
        
        # --- CLASSIFICATION DE DIFFICULTÉ ---
        difficulty_data = df['gradient'].apply(get_piste_difficulty)
        df['color_name'] = difficulty_data.apply(lambda x: x[0])
        df['hex_color'] = difficulty_data.apply(lambda x: x[1])
        df['difficulty_score'] = difficulty_data.apply(lambda x: x[2])
        
        # --- DÉTECTION D'ÉTAT (Ski / Remontée / Arrêt) ---
        df['state'] = 'Arrêt'
        
        # Remontée mécanique : montée + vitesse moyenne
        df.loc[
            (df['ele_diff'] > 1) & 
            (df['speed_kmh'] > 3) & 
            (df['speed_kmh'] < 30),
            'state'
        ] = 'Remontée'
        
        # Descente : perte d'altitude + vitesse
        df.loc[
            (df['ele_diff'] < -0.5) & 
            (df['speed_kmh'] > 5),
            'state'
        ] = 'Ski'
        
        # --- MÉTRIQUES CUMULATIVES ---
        df['cumulative_dist'] = df['dist'].cumsum() / 1000  # en km
        df['cumulative_descent'] = df['ele_diff'].clip(upper=0).abs().cumsum()
        df['cumulative_ascent'] = df['ele_diff'].clip(lower=0).cumsum()
        
        # RGB pour visualisation
        df['color_rgb'] = df['hex_color'].apply(
            lambda x: [int(x[1:3], 16), int(x[3:5], 16), int(x[5:7], 16)]
        )
        
        return df
    
    except Exception as e:
        raise Exception(f"Erreur lors du parsing GPX : {str(e)}")

def detect_runs(df):
    """
    Détecte et analyse les descentes individuelles
    Returns:
        DataFrame avec une ligne par descente
    """
    # Segmentation basée sur le changement d'état
    df['segment_id'] = (df['state'] != df['state'].shift()).cumsum()
    
    runs = []
    run_number = 1
    
    for seg_id, group in df.groupby('segment_id'):
        if group['state'].iloc[0] != 'Ski':
            continue
        
        # Filtres de qualité
        duration = group['dt'].sum()
        drop = group['ele'].max() - group['ele'].min()
        distance = group['dist'].sum()
        
        if duration < MIN_RUN_DURATION or drop < MIN_RUN_DROP:
            continue
        
        # Analyse détaillée de la descente
        avg_speed = group['speed_kmh'].mean()
        max_speed = group['speed_kmh'].max()
        avg_gradient = group['gradient'].abs().mean()
        max_gradient = group['gradient'].abs().max()
        
        # Couleur dominante (mode)
        dominant_color = group['color_name'].mode()[0] if not group['color_name'].mode().empty else "Inconnue"
        
        # Répartition des couleurs
        color_distribution = group['color_name'].value_counts(normalize=True) * 100
        
        # Point central pour OSM
        mid_idx = len(group) // 2
        mid_point = group.iloc[mid_idx]
        
        # Coordonnées de départ et arrivée
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
    
    # Trace pour chaque segment de couleur
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

def create_3d_map(df):
    """Crée une carte 3D interactive avec PyDeck"""
    # Filtrer uniquement les descentes
    ski_data = df[df['state'] == 'Ski'].copy()
    
    if ski_data.empty:
        return None
    
    # Layer de trajectoire
    layer = pdk.Layer(
        "PathLayer",
        data=ski_data,
        pickable=True,
        get_path=ski_data[['lon', 'lat']].values.tolist(),
        get_color='color_rgb',
        width_scale=20,
        width_min_pixels=3,
        get_width=5,
    )
    
    # Points de départ/arrivée
    start_end_layer = pdk.Layer(
        "ScatterplotLayer",
        data=ski_data.iloc[[0, -1]],
        get_position=['lon', 'lat'],
        get_color=[0, 255, 0, 200],
        get_radius=50,
        pickable=True,
    )
    
    # Vue centrée
    view_state = pdk.ViewState(
        latitude=ski_data['lat'].mean(),
        longitude=ski_data['lon'].mean(),
        zoom=13,
        pitch=60,
        bearing=0
    )
    
    # Deck
    r = pdk.Deck(
        layers=[layer, start_end_layer],
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

# --- INTERFACE STREAMLIT ---

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/skiing.png", width=80)
    st.title("⚙️ Paramètres")
    
    st.markdown("---")
    st.subheader("Filtres de détection")
    
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
    st.subheader("Visualisation")
    
    show_3d = st.checkbox("Afficher carte 3D", value=True)
    show_stats = st.checkbox("Statistiques détaillées", value=True)
    
    st.markdown("---")
    st.info("""
    **💡 Astuce :**
    - Exportez vos traces en GPX depuis Slopes
    - Les noms de pistes sont détectés via OpenStreetMap
    - Utilisez Ctrl+Clic pour pivoter la vue 3D
    """)

# Interface principale
st.title("⛷️ Ski Analytics Pro - Édition IA")
st.markdown("""
Analysez automatiquement vos sessions de ski : détection de difficulté, 
identification des pistes et statistiques avancées.
""")

# Upload de fichier
uploaded_file = st.file_uploader(
    "📂 Importez votre fichier GPX",
    type=['gpx', 'xml', 'txt'],
    help="Formats acceptés : GPX, XML ou TXT contenant des données GPS"
)

if uploaded_file is not None:
    try:
        with st.spinner("🔄 Analyse du fichier en cours..."):
            # Lecture du fichier
            file_content = uploaded_file.read()
            
            # Tentative de décodage
            try:
                file_content = file_content.decode('utf-8')
            except:
                file_content = file_content.decode('latin-1')
            
            # Parse et enrichissement
            df = parse_and_enrich_gpx(file_content)
            
            # Mise à jour des paramètres
            MIN_RUN_DURATION = min_duration
            MIN_RUN_DROP = min_drop
            
            # Détection des descentes
            runs_df = detect_runs(df)
        
        st.success(f"✅ Fichier '{uploaded_file.name}' analysé avec succès !")
        
        # --- SECTION 1 : STATISTIQUES GLOBALES ---
        st.markdown("### 📊 Vue d'ensemble de la session")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🏔️ Altitude Max",
                f"{int(df['ele'].max())} m"
            )
        
        with col2:
            st.metric(
                "⬇️ Dénivelé Total",
                f"{int(df['cumulative_descent'].max())} m"
            )
        
        with col3:
            st.metric(
                "🛷 Descentes",
                len(runs_df)
            )
        
        with col4:
            st.metric(
                "🚀 Vitesse Max",
                f"{df['speed_kmh'].max():.1f} km/h"
            )
        
        with col5:
            total_ski_time = df[df['state'] == 'Ski']['dt'].sum()
            st.metric(
                "⏱️ Temps Ski",
                f"{int(total_ski_time//60)} min"
            )
        
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
        
        # --- SECTION 3 : TABLEAU DES DESCENTES ---
        st.markdown("### 🎿 Détail des Descentes")
        
        if not runs_df.empty:
            # Option de recherche OSM
            col_osm, col_export = st.columns([3, 1])
            
            with col_osm:
                identify_names = st.checkbox(
                    "🔍 Identifier les pistes via OpenStreetMap (peut prendre du temps)",
                    help="Interroge la base OpenStreetMap pour trouver les noms de pistes"
                )
            
            with col_export:
                csv_data = export_to_csv(runs_df)
                st.download_button(
                    label="📥 Exporter CSV",
                    data=csv_data,
                    file_name=f"ski_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
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
            
            # Fonction de coloration
            def color_difficulty(val):
                colors = {
                    'Verte': 'background-color: #90EE90; color: black; font-weight: bold',
                    'Bleue': 'background-color: #ADD8E6; color: black; font-weight: bold',
                    'Rouge': 'background-color: #FFcccb; color: black; font-weight: bold',
                    'Noire': 'background-color: #D3D3D3; color: black; font-weight: bold'
                }
                return colors.get(val, '')
            
            # Affichage du tableau stylisé
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
            
            # Statistiques détaillées si activé
            if show_stats:
                st.markdown("### 📈 Statistiques Avancées")
                
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                with stat_col1:
                    st.markdown("**Distribution des Pistes**")
                    color_counts = runs_df['Couleur Dominante'].value_counts()
                    for color, count in color_counts.items():
                        st.write(f"{color} : {count} descentes")
                
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
        
        # --- SECTION 4 : CARTE 3D ---
        if show_3d:
            st.markdown("### 🗺️ Visualisation 3D Interactive")
            
            deck = create_3d_map(df)
            if deck:
                st.pydeck_chart(deck)
                st.info("💡 Maintenez Ctrl + Clic pour faire pivoter la vue en 3D")
            else:
                st.warning("Aucune donnée de descente à afficher")
        
        # Détails par descente (expandable)
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
                
                # Graphique spécifique à la descente
                run_points = run_data['points']
                
                fig_run = go.Figure()
                fig_run.add_trace(go.Scatter(
                    x=run_points['time'],
                    y=run_points['speed_kmh'],
                    name='Vitesse',
                    line=dict(color='#667eea', width=2)
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
        st.error(f"❌ Erreur lors​​​​​​​​​​​​​​​​
