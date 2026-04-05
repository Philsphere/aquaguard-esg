import streamlit as st
import requests
import folium
from folium import plugins
from streamlit_folium import st_folium
import math
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- PAGE SETUP ---
st.set_page_config(page_title="AquaGuard ESG Auditor 🛰️", layout="wide", initial_sidebar_state="expanded")

# --- INITIAL STATE ---
if 'lat' not in st.session_state: st.session_state.lat = 36.8340
if 'lon' not in st.session_state: st.session_state.lon = -2.4637
if 'address_name' not in st.session_state: st.session_state.address_name = "Almería, Andalusien, Spanien"
if 'search_geojson' not in st.session_state: st.session_state.search_geojson = None 

# --- FUNCTIONS ---
@st.cache_data(ttl=3600)
def geocode_pro(query):
    url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1&polygon_geojson=1"
    headers = {'User-Agent': 'AquaGuard-Enterprise/1.0'}
    try:
        res = requests.get(url, headers=headers).json()
        if res: 
            data = res[0]
            geojson_data = data.get('geojson', None)
            return float(data['lat']), float(data['lon']), data['display_name'], geojson_data
    except: pass
    return None, None, None, None

def simulate_field_vitality(f_lat, f_lon):
    seed = int((abs(f_lat) + abs(f_lon)) * 1000000) % 10
    if seed > 6: 
        return 0.85, True 
    else: 
        return 0.35 + (seed * 0.02), False

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛰️ AquaGuard ESG")
    st.caption("Copernicus Sentinel-2 & ERA5 Data Fusion")
    st.markdown("---")
    
    mode = st.radio("System-Modus", ["🌍 Global Supplier Radar (Makro)", "🎯 Batch-Auditor (Mikro)"])
    st.markdown("---")
    
    st.markdown("### 📍 Audit-Location")
    search_q = st.text_input("Lieferanten-Adresse", placeholder="z.B. Almería, Spanien")
    if st.button("Suchen & Zentrieren", use_container_width=True):
        if search_q:
            lat, lon, name, geojson = geocode_pro(search_q)
            if lat:
                st.session_state.lat, st.session_state.lon, st.session_state.address_name = lat, lon, name
                st.session_state.search_geojson = geojson
                st.success("Target locked.")
                # Auto-Switch to Micro Mode on search
                mode = "🎯 Batch-Auditor (Mikro)"
            else: st.error("Target nicht gefunden.")

    st.caption(f"Fokus: {st.session_state.address_name.split(',')[0]}")
    st.markdown("---")
    
    if mode == "🌍 Global Supplier Radar (Makro)":
        risk_sens = st.slider("Sensibilität für Wasserraub-Anomalien", 10, 80, 50)
        st.info("💡 Zeigt Tier-1 Zulieferer aus der internen Datenbank. Rote Knotenpunkte erfordern einen sofortigen Mikro-Audit.")
        
    elif mode == "🎯 Batch-Auditor (Mikro)":
        water_rights = st.selectbox("Registrierte Wasserlizenz", ["Keine Lizenz gefunden (Illegal)", "Limitierte Lizenz (500m³/ha)", "Volle Lizenz (Legal)"])
        st.info("💡 Zeichne **mehrere Felder** eines Lieferanten ein. Das System führt einen simultanen Batch-Scan durch.")

# --- MACRO MODUS (SUPPLIER RADAR) ---
if mode == "🌍 Global Supplier Radar (Makro)":
    st.title("🌐 Corporate Supply Chain: ESG Risk Radar")
    st.markdown("Globales Monitoring aller registrierten Tier-1 Agrar-Zulieferer. Datenfusion aus ERA5-Dürreindizes und Sentinel-Anomalien.")
    
    # Fake Supplier Database
    suppliers = pd.DataFrame({
        'Supplier_ID': ['SUP-001', 'SUP-002', 'SUP-003', 'SUP-004', 'SUP-005', 'SUP-006', 'SUP-007', 'SUP-008', 'SUP-009'],
        'Name': ['Almería Greenhouses S.A.', 'Nevada Desert Farms LLC', 'Moroccan Citrus Coop', 'Bavarian Wheat Corp', 'Central Valley Almonds', 'Cape Town Vineyards', 'Mato Grosso Soy', 'Puglia Olive Oil', 'Nile Delta Cotton'],
        'Lat': [36.8340, 36.1699, 30.4278, 48.1351, 36.7468, -33.9249, -12.6819, 41.1171, 30.8025],
        'Lon': [-2.4637, -115.1398, -9.5981, 11.5820, -119.7726, 18.4241, -56.9211, 16.8719, 31.0218],
        'Rohstoff': ['Tomaten', 'Alfalfa', 'Zitrusfrüchte', 'Weizen', 'Mandeln', 'Trauben', 'Soja', 'Oliven', 'Baumwolle'],
        'Umsatzvolumen_Mio': [12.5, 8.2, 4.1, 15.0, 22.4, 5.6, 35.0, 9.1, 7.8]
    })
    
    # Dynamisches Risiko generieren (Sensibilität beeinflusst die Anzahl der roten Alerts)
    np.random.seed(42) # Für stabile Demo-Werte
    base_risks = np.random.randint(10, 60, size=len(suppliers))
    # Hotspots künstlich erhöhen
    base_risks[0] += 40 # Almeria
    base_risks[1] += 35 # Nevada
    suppliers['ESG_Risk_Score'] = np.clip(base_risks + (risk_sens - 50) * 0.5, 0, 100)
    
    # Status festlegen
    suppliers['Status'] = ['Kritisch (Audit nötig)' if r > 75 else 'Erhöhtes Risiko' if r > 50 else 'Compliant' for r in suppliers['ESG_Risk_Score']]
    
    c1, c2, c3 = st.columns(3)
    critical_count = len(suppliers[suppliers['Status'] == 'Kritisch (Audit nötig)'])
    c1.metric("Überwachte Tier-1 Lieferanten", len(suppliers))
    c2.metric("Kritische ESG-Anomalien", critical_count, f"{critical_count} offene Audits", delta_color="inverse")
    c3.metric("Value at Risk (CSRD Bußgelder)", f"~ {critical_count * 2.5} Mio €", "Geschätzt")
    
    with st.spinner("Synchronisiere globale Satelliten-Knoten..."):
        fig = px.scatter_mapbox(
            suppliers, 
            lat="Lat", lon="Lon", 
            hover_name="Name", 
            hover_data={"Lat": False, "Lon": False, "Rohstoff": True, "Umsatzvolumen_Mio": True, "ESG_Risk_Score": True, "Status": True},
            color="ESG_Risk_Score",
            size="Umsatzvolumen_Mio",
            color_continuous_scale=px.colors.diverging.RdYlGn[::-1], # Rot ist schlecht, Grün ist gut
            range_color=[0, 100],
            zoom=1.5,
            mapbox_style="carto-darkmatter"
        )
        
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117"
        )
        st.plotly_chart(fig, use_container_width=True, height=600)
        
        st.markdown("### 📋 Kritische Lieferanten (Watchlist)")
        st.dataframe(suppliers[suppliers['Status'] != 'Compliant'].sort_values('ESG_Risk_Score', ascending=False), use_container_width=True, hide_index=True)

# --- MICRO MODUS (BATCH AUDITOR) ---
elif mode == "🎯 Batch-Auditor (Mikro)":
    st.title("🎯 Data Fusion: Multi-Field Anomaly Detection")
    st.markdown(f"**Audit Ziel:** {st.session_state.address_name}")
    
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=15, tiles=None)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Satellit').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Straßenkarte').add_to(m)
    
    if st.session_state.search_geojson:
        folium.GeoJson(st.session_state.search_geojson, name="Audit Zone", style_function=lambda x: {'fillColor': '#e74c3c', 'color': '#e74c3c', 'weight': 3, 'fillOpacity': 0.3}).add_to(m)
    
    draw = plugins.Draw(export=False, draw_options={'polyline':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':False})
    draw.add_to(m)
    folium.LayerControl().add_to(m)
    
    map_out = st_folium(m, use_container_width=True, height=450, returned_objects=['all_drawings'])

    if map_out and map_out.get('all_drawings') and len(map_out['all_drawings']) > 0:
        drawings = map_out['all_drawings']
        
        with st.spinner(f"Führe Batch-Scan für {len(drawings)} Feld(er) durch..."):
            base_coords = drawings[0]['geometry']['coordinates'][0]
            center_lat = np.mean([c[1] for c in base_coords])
            center_lon = np.mean([c[0] for c in base_coords])
            
            try:
                url = f"https://api.open-meteo.com/v1/forecast?latitude={center_lat}&longitude={center_lon}&daily=precipitation_sum,et0_fao_evapotranspiration&timezone=Europe/Berlin&past_days=90&forecast_days=0"
                data = requests.get(url).json()['daily']
                regen = [r if r is not None else 0 for r in data['precipitation_sum']]
                et0 = [e if e is not None else 0 for e in data['et0_fao_evapotranspiration']]
                
                natural_moisture = 100.0
                moisture_history = []
                for r, e in zip(regen, et0):
                    natural_moisture = max(0, natural_moisture + r - (e * 1.0))
                    moisture_history.append(natural_moisture)
                
                current_moisture = moisture_history[-1]
                times = pd.to_datetime(data['time'])
                
                audit_results = []
                total_stolen_water = 0
                worst_field_id = 0
                highest_ndvi = 0
                
                for i, polygon in enumerate(drawings):
                    coords = polygon['geometry']['coordinates'][0]
                    f_lat, f_lon = np.mean([c[1] for c in coords]), np.mean([c[0] for c in coords])
                    
                    lat_to_m = 111320.0
                    lon_to_m = 40075000.0 * math.cos(math.radians(f_lat)) / 360.0
                    area = 0.0
                    for j in range(len(coords) - 1):
                        area += (coords[j][0] * lon_to_m * coords[j+1][1] * lat_to_m) - (coords[j+1][0] * lon_to_m * coords[j][1] * lat_to_m)
                    netto_ha = abs(area) / 20000.0 * 0.98 
                    
                    current_ndvi, is_irrigated = simulate_field_vitality(f_lat, f_lon)
                    is_anomaly = current_moisture < 40 and is_irrigated
                    stolen_water_m3 = netto_ha * 10000 * (40 - current_moisture) / 1000 if is_anomaly else 0
                    
                    if "Keine Lizenz" in water_rights and is_anomaly:
                        status = "ILLEGAL (Verstoß EU-CSRD)"
                        total_stolen_water += stolen_water_m3
                        if current_ndvi > highest_ndvi:
                            highest_ndvi = current_ndvi
                            worst_field_id = i + 1
                    elif is_anomaly:
                        status = "LEGAL (Lizenz vorhanden)"
                    else:
                        status = "LEGAL (Natürlicher Zustand)"

                    audit_results.append({
                        "Audit ID": f"AQG-{datetime.now().strftime('%Y%m%d')}-{i+1}",
                        "Koordinaten (Lat/Lon)": f"{f_lat:.4f}, {f_lon:.4f}",
                        "Fläche (ha)": round(netto_ha, 2),
                        "Satelliten-NDVI": round(current_ndvi, 2),
                        "Bodenwasser (mm)": round(current_moisture, 1),
                        "Unerklärte Menge (m³)": round(stolen_water_m3, 0),
                        "Compliance Status": status
                    })

                st.markdown("---")
                st.markdown(f"### 📋 Automatischer Audit-Report")
                
                df_results = pd.DataFrame(audit_results)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Geprüfte Felder", len(drawings))
                c2.metric("Illegale Entnahme (Gesamt)", f"{total_stolen_water:,.0f} m³", delta_color="inverse" if total_stolen_water > 0 else "normal")
                c3.metric("Potenzielles Bußgeld", f"{total_stolen_water * 1.50:,.0f} €", delta_color="inverse" if total_stolen_water > 0 else "normal")
                c4.metric("Worst Offender", f"Flurstück {worst_field_id}" if total_stolen_water > 0 else "Keiner")

                def color_status(val):
                    color = '#ff4b4b' if 'ILLEGAL' in str(val) else '#20c997' if 'LEGAL' in str(val) else 'white'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(df_results.style.map(color_status, subset=['Compliance Status']), use_container_width=True, hide_index=True)
                
                st.markdown("#### 📥 Beweissicherung Exportieren")
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Offizielles Audit-Log herunterladen (CSV)",
                    data=csv,
                    file_name=f"AquaGuard_Audit_{st.session_state.address_name.split(',')[0].replace(' ', '_')}.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                if total_stolen_water > 0:
                    st.markdown(f"**Detail-Analyse: Flurstück {worst_field_id} (Höchste Anomalie)**")
                    ndvi_history = [highest_ndvi + np.random.normal(0, 0.01) for _ in range(len(moisture_history))]
                else:
                    st.markdown("**Detail-Analyse: Flurstück 1 (Referenz)**")
                    ref_ndvi = audit_results[0]["Satelliten-NDVI"]
                    ndvi_history = [ref_ndvi + np.random.normal(0, 0.01) for _ in range(len(moisture_history))]

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=times, y=moisture_history, name="Natürliches Bodenwasser (ERA5)", line=dict(color="#e74c3c", width=3), fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'), secondary_y=False)
                fig.add_trace(go.Scatter(x=times, y=ndvi_history, name="Satelliten Vitalität (NDVI)", line=dict(color="#2ecc71", width=3)), secondary_y=True)
                
                if total_stolen_water > 0:
                    fig.add_vrect(x0=times[-30], x1=times[-1], fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="ANOMALIE (Irrigation)", annotation_position="top left")

                fig.update_layout(
                    title="Data Fusion Profil", template="plotly_dark", hovermode="x unified",
                    xaxis_title="Datum", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_yaxes(title_text="Bodenwasser (mm)", secondary_y=False, color="#e74c3c")
                fig.update_yaxes(title_text="Vegetations-Index (NDVI)", secondary_y=True, color="#2ecc71", range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ System-Fehler bei der API-Verarbeitung: {str(e)}")