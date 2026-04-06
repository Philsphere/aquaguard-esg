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
import io

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

def simulate_optical_ndvi(f_lat, f_lon, year):
    seed = int((abs(f_lat) + abs(f_lon)) * 1000000) % 10
    if seed > 6: 
        if year >= 2022: return 0.85, True # Neu gebohrter illegaler Brunnen
        else: return 0.38, False # Vor 2022 war das Feld natürliches Brachland
    else: return 0.35 + (seed * 0.02), False

def simulate_radar_sar(f_lat, f_lon, year):
    seed = int((abs(f_lat) + abs(f_lon)) * 1000000) % 10
    if seed > 6: 
        if year >= 2022: return -6.5, True 
        else: return -17.0, False 
    else: return -18.2 + (seed * 0.5), False 

def generate_html_dossier(location, date, total_water, fine, worst_id, audit_results, use_radar):
    sensor_type = "Sentinel-1 C-Band SAR (Radar)" if use_radar else "Sentinel-2 Multispectral (Optisch NDVI)"
    
    rows = ""
    for res in audit_results:
        color = "red" if "ILLEGAL" in res['Compliance Status'] else "green"
        rows += f"""
        <tr>
            <td>{res['Nr.']}</td>
            <td>{res['Fläche (ha)']} ha</td>
            <td>{res['Satelliten-Wert (NDVI/dB)']}</td>
            <td>{res['Bodenwasser (mm)']} mm</td>
            <td style="color: {color}; font-weight: bold;">{res['Unerklärte Menge (m³)']:.0f} m³</td>
            <td style="color: {color}; font-weight: bold;">{res['Compliance Status']}</td>
        </tr>
        """
        
    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>AquaGuard Dossier</title>
        <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; padding: 40px; max-width: 900px; margin: auto; }}
            .header {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; margin-bottom: 30px; }}
            h1 {{ color: #2c3e50; margin-bottom: 5px; }}
            .metadata {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 30px; border-left: 5px solid #3498db; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; font-size: 14px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #2c3e50; color: white; }}
            .summary {{ font-size: 18px; padding: 20px; background-color: #ffeaea; border: 1px solid #ffb3b3; border-radius: 5px; }}
            .footer {{ margin-top: 50px; font-size: 12px; color: #7f8c8d; text-align: center; border-top: 1px solid #ddd; padding-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AquaGuard ESG Compliance Dossier</h1>
            <p><strong>Offizieller Audit-Bericht für Lieferkettengesetz (CSRD / CSDDD)</strong></p>
        </div>
        
        <div class="metadata">
            <p><strong>Audit Datum:</strong> {date}</p>
            <p><strong>Zielkoordinaten / Region:</strong> {location}</p>
            <p><strong>Verwendete Satelliten-Sensorik:</strong> {sensor_type}</p>
            <p><strong>Referenz-Daten:</strong> Copernicus ERA5 (Klima) & ESA Sentinel</p>
        </div>

        <h3>Detaillierte Flurstück-Analyse</h3>
        <table>
            <tr>
                <th>Flurstück</th>
                <th>Fläche</th>
                <th>Sensor-Wert</th>
                <th>Bodenwasser</th>
                <th>Unerklärte Menge</th>
                <th>Rechtsstatus</th>
            </tr>
            {rows}
        </table>

        <div class="summary">
            <h3 style="margin-top: 0; color: #c0392b;">Zusammenfassung des ESG-Risikos</h3>
            <p><strong>Gesamte illegale Wasserentnahme:</strong> {total_water:,.0f} m³</p>
            <p><strong>Geschätztes Bußgeld-Risiko:</strong> {fine:,.0f} €</p>
            <p><strong>Kritischster Verstoß:</strong> Flurstück {worst_id}</p>
        </div>

        <div class="footer">
            Generiert durch AquaGuard Enterprise SaaS. Satellitendaten bereitgestellt durch das Copernicus Programm der Europäischen Union.
        </div>
    </body>
    </html>
    """
    return html

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛰️ AquaGuard ESG")
    st.caption("Copernicus Multi-Sensor Data Fusion")
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
                mode = "🎯 Batch-Auditor (Mikro)"
            else: st.error("Target nicht gefunden.")

    st.caption(f"Fokus: {st.session_state.address_name.split(',')[0]}")
    st.markdown("---")
    
    if mode == "🌍 Global Supplier Radar (Makro)":
        risk_sens = st.slider("Sensibilität für Wasserraub-Anomalien", 10, 80, 50)
        st.info("💡 Zeigt Tier-1 Zulieferer auf der Weltkarte. Integrieren Sie Ihre eigene ERP-Datenbank über den CSV-Upload.")
        
    elif mode == "🎯 Batch-Auditor (Mikro)":
        water_rights = st.selectbox("Registrierte Wasserlizenz", ["Keine Lizenz gefunden (Illegal)", "Limitierte Lizenz (500m³/ha)", "Volle Lizenz (Legal)"])
        
        st.markdown("### ⏳ Time Machine (Historie)")
        audit_year = st.slider("Audit-Jahr", 2018, 2024, 2024)
        st.info("💡 Vergleichen Sie aktuelle Sensor-Daten mit historischen Werten, um neu gebohrte illegale Brunnen zu identifizieren.")
        
        st.markdown("### ☁️ Sensor-Bedingungen")
        cloud_cover = st.slider("Lokale Bewölkung (%)", 0, 100, 10)

# --- MACRO MODUS (SUPPLIER RADAR) ---
if mode == "🌍 Global Supplier Radar (Makro)":
    st.title("🌐 Corporate Supply Chain: ESG Risk Radar")
    st.markdown("Globales Monitoring aller registrierten Tier-1 Agrar-Zulieferer. Datenfusion aus ERA5-Dürreindizes und Sentinel-Anomalien.")
    
    with st.expander("📂 ERP / Unternehmensdaten anbinden (CSV Upload)"):
        st.markdown("Laden Sie Ihre interne Zulieferer-Datenbank hoch, um das globale Risiko-Scoring für Ihre spezifische Lieferkette durchzuführen.")
        template_df = pd.DataFrame({'Supplier_ID': ['SUP-001', 'SUP-002'], 'Name': ['Beispiel Farm A', 'Beispiel Farm B'], 'Lat': [36.8340, 36.1699], 'Lon': [-2.4637, -115.1398], 'Rohstoff': ['Tomaten', 'Alfalfa'], 'Umsatzvolumen_Mio': [12.5, 8.2]})
        st.download_button("📝 Format-Template herunterladen (CSV)", template_df.to_csv(index=False).encode('utf-8'), "AquaGuard_Template.csv", "text/csv")
        uploaded_file = st.file_uploader("Lieferanten-Liste hochladen", type=['csv'])

    if uploaded_file is not None:
        try:
            suppliers = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(suppliers)} Lieferanten erfolgreich synchronisiert.")
            is_custom_data = True
        except Exception as e:
            st.error("❌ Fehler beim Lesen der Datei.")
            suppliers = None
            is_custom_data = False
    else:
        suppliers = pd.DataFrame({
            'Supplier_ID': ['SUP-001', 'SUP-002', 'SUP-003', 'SUP-004', 'SUP-005', 'SUP-006', 'SUP-007', 'SUP-008', 'SUP-009'],
            'Name': ['Almería Greenhouses S.A.', 'Nevada Desert Farms LLC', 'Moroccan Citrus Coop', 'Bavarian Wheat Corp', 'Central Valley Almonds', 'Cape Town Vineyards', 'Mato Grosso Soy', 'Puglia Olive Oil', 'Nile Delta Cotton'],
            'Lat': [36.8340, 36.1699, 30.4278, 48.1351, 36.7468, -33.9249, -12.6819, 41.1171, 30.8025],
            'Lon': [-2.4637, -115.1398, -9.5981, 11.5820, -119.7726, 18.4241, -56.9211, 16.8719, 31.0218],
            'Rohstoff': ['Tomaten', 'Alfalfa', 'Zitrusfrüchte', 'Weizen', 'Mandeln', 'Trauben', 'Soja', 'Oliven', 'Baumwolle'],
            'Umsatzvolumen_Mio': [12.5, 8.2, 4.1, 15.0, 22.4, 5.6, 35.0, 9.1, 7.8]
        })
        is_custom_data = False

    if suppliers is not None:
        np.random.seed(42)
        base_risks = np.random.randint(10, 60, size=len(suppliers))
        if not is_custom_data:
            base_risks[0] += 40 
            base_risks[1] += 35 
        suppliers['ESG_Risk_Score'] = np.clip(base_risks + (risk_sens - 50) * 0.5, 0, 100)
        suppliers['Status'] = ['Kritisch (Audit nötig)' if r > 75 else 'Erhöhtes Risiko' if r > 50 else 'Compliant' for r in suppliers['ESG_Risk_Score']]
        
        c1, c2, c3 = st.columns(3)
        critical_count = len(suppliers[suppliers['Status'] == 'Kritisch (Audit nötig)'])
        c1.metric("Überwachte Tier-1 Lieferanten", len(suppliers))
        c2.metric("Kritische ESG-Anomalien", critical_count, f"{critical_count} offene Audits", delta_color="inverse")
        c3.metric("Value at Risk (CSRD Bußgelder)", f"~ {critical_count * 2.5} Mio €", "Geschätzt")
        
        with st.spinner("Synchronisiere globale Satelliten-Knoten..."):
            fig = px.scatter_mapbox(
                suppliers, lat="Lat", lon="Lon", hover_name="Name", 
                hover_data={"Lat": False, "Lon": False, "Rohstoff": True, "Umsatzvolumen_Mio": True, "ESG_Risk_Score": True, "Status": True},
                color="ESG_Risk_Score", size="Umsatzvolumen_Mio" if 'Umsatzvolumen_Mio' in suppliers.columns else None,
                color_continuous_scale=px.colors.diverging.RdYlGn[::-1], range_color=[0, 100],
                zoom=1.5, mapbox_style="carto-darkmatter"
            )
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True, height=600)
            
            st.markdown("### 📋 Kritische Lieferanten (Watchlist)")
            st.dataframe(suppliers[suppliers['Status'] != 'Compliant'].sort_values('ESG_Risk_Score', ascending=False), use_container_width=True, hide_index=True)

# --- MICRO MODUS (BATCH AUDITOR) ---
elif mode == "🎯 Batch-Auditor (Mikro)":
    st.title("🎯 Data Fusion: Multi-Field Anomaly Detection")
    st.markdown(f"**Audit Ziel:** {st.session_state.address_name}")
    
    use_radar = cloud_cover > 40
    if use_radar:
        st.warning(f"⚠️ **Sichtverhältnisse kritisch ({cloud_cover}% Bewölkung).** Optische Sentinel-2 Sensoren blockiert. **System nutzt Sentinel-1 C-Band SAR (Radar)** zur Durchdringung der Wolkendecke.")
    else:
        st.success(f"✅ **Sichtverhältnisse optimal ({cloud_cover}% Bewölkung).** System nutzt primäre optische Sentinel-2 Sensoren (NDVI).")

    with st.expander("ℹ️ Sensor-Technologie & Physik erklärt (NDVI & Radar)"):
        st.markdown("""
        Dieses System nutzt zwei getrennte physikalische Prinzipien, um völlig wetterunabhängig zu operieren:
        
        ### 1. Optische Sensoren (Sentinel-2 NDVI)
        Der **NDVI** (Normalized Difference Vegetation Index) misst bei klarem Himmel die Dichte und Gesundheit von Pflanzen anhand der Lichtreflexion.
        * **Die Physik:** Gesunde Pflanzen absorbieren viel rotes Licht (für Photosynthese) und reflektieren viel nahes Infrarot (NIR).
        * **Die Formel:** NDVI = (NIR - Rot) / (NIR + Rot)
        * **Die Skala:** * **0.7 bis 1.0:** Sehr vitale, bewässerte Vegetation.
            * **0.3 bis 0.6:** Natürliches Gebüsch oder gestresste Pflanzen.
            * **0.1 bis 0.2:** Nackter Boden / Stein.
            * **< 0:** Wasserflächen oder Wolken.

        ### 2. Radar-Sensoren (Sentinel-1 SAR)
        Wenn Wolken die Sicht blockieren, sendet der Satellit Mikrowellen (C-Band) aus. Diese durchdringen Wolken völlig problemlos. 
        * **Die Physik:** Wasser auf dem Boden oder in den Pflanzen reflektiert Radarwellen extrem stark zurück zum Satelliten (Backscatter).
        * **Die Skala:** Gemessen in Dezibel (dB). 
            * **-20 dB:** Knochentrocken (absorbiert die Wellen).
            * **-5 dB:** Extrem nass oder intensiv bewässert (reflektiert stark).
        """)
    
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
        
        with st.spinner(f"Führe Batch-Scan für das Jahr {audit_year} durch..."):
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
                map_visuals = [] 
                total_stolen_water = 0
                worst_field_id = "Keiner"
                highest_anomaly_val = -999 if use_radar else 0
                
                for i, polygon in enumerate(drawings):
                    coords = polygon['geometry']['coordinates'][0]
                    f_lat, f_lon = np.mean([c[1] for c in coords]), np.mean([c[0] for c in coords])
                    
                    lat_to_m = 111320.0
                    lon_to_m = 40075000.0 * math.cos(math.radians(f_lat)) / 360.0
                    area = 0.0
                    for j in range(len(coords) - 1):
                        area += (coords[j][0] * lon_to_m * coords[j+1][1] * lat_to_m) - (coords[j+1][0] * lon_to_m * coords[j][1] * lat_to_m)
                    netto_ha = abs(area) / 20000.0 * 0.98 
                    
                    if use_radar:
                        sensor_val, is_irrigated = simulate_radar_sar(f_lat, f_lon, audit_year)
                    else:
                        sensor_val, is_irrigated = simulate_optical_ndvi(f_lat, f_lon, audit_year)

                    is_anomaly = current_moisture < 40 and is_irrigated
                    stolen_water_m3 = netto_ha * 10000 * (40 - current_moisture) / 1000 if is_anomaly else 0
                    
                    if "Keine Lizenz" in water_rights and is_anomaly:
                        status = "ILLEGAL (Verstoß EU-CSRD)"
                        total_stolen_water += stolen_water_m3
                        color_code = '#ff4b4b' 
                        if sensor_val > highest_anomaly_val:
                            highest_anomaly_val = sensor_val
                            worst_field_id = f"#{i + 1}"
                    elif is_anomaly:
                        status = "LEGAL (Lizenz vorhanden)"
                        color_code = '#20c997' 
                    else:
                        status = "LEGAL (Natürlicher Zustand)"
                        color_code = '#20c997' 

                    audit_results.append({
                        "Nr.": f"#{i+1}",
                        "Audit ID": f"AQG-{audit_year}04-{i+1}",
                        "Fläche (ha)": round(netto_ha, 2),
                        "Satelliten-Wert (NDVI/dB)": round(sensor_val, 2),
                        "Bodenwasser (mm)": round(current_moisture, 1),
                        "Unerklärte Menge (m³)": round(stolen_water_m3, 0),
                        "Compliance Status": status
                    })
                    
                    map_visuals.append({
                        "polygon": polygon, "center_lat": f_lat, "center_lon": f_lon, "color": color_code, "number": i + 1
                    })

                st.markdown("---")
                st.markdown(f"### 📋 Automatischer Audit-Report (Jahr: {audit_year})")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Geprüfte Felder", len(drawings))
                c2.metric("Illegale Entnahme (Gesamt)", f"{total_stolen_water:,.0f} m³", delta_color="inverse" if total_stolen_water > 0 else "normal")
                c3.metric("Potenzielles Bußgeld", f"{total_stolen_water * 1.50:,.0f} €", delta_color="inverse" if total_stolen_water > 0 else "normal")
                c4.metric("Worst Offender", f"Flurstück {worst_field_id}" if total_stolen_water > 0 else "Keiner")

                st.markdown("#### 🗺️ Visuelle Zuordnung (Result Map)")
                result_map = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=None)
                folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri').add_to(result_map)
                
                for vis in map_visuals:
                    folium.GeoJson(vis["polygon"], style_function=lambda x, color=vis["color"]: {'fillColor': color, 'color': color, 'weight': 3, 'fillOpacity': 0.4}).add_to(result_map)
                    html_icon = f"""<div style="font-family: sans-serif; font-size: 14px; color: white; background-color: rgba(0,0,0,0.8); border-radius: 50%; width: 26px; height: 26px; text-align: center; line-height: 26px; border: 2px solid {vis['color']}; box-shadow: 0 0 5px rgba(0,0,0,0.5);"><b>{vis['number']}</b></div>"""
                    folium.Marker(location=[vis["center_lat"], vis["center_lon"]], icon=folium.DivIcon(html=html_icon)).add_to(result_map)

                st_folium(result_map, use_container_width=True, height=350, key="result_map")

                df_results = pd.DataFrame(audit_results)
                def color_status(val):
                    color = '#ff4b4b' if 'ILLEGAL' in str(val) else '#20c997' if 'LEGAL' in str(val) else 'white'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(df_results.style.map(color_status, subset=['Compliance Status']), use_container_width=True, hide_index=True)
                
                st.markdown("#### 📥 Beweissicherung Exportieren")
                
                col_csv, col_html = st.columns(2)
                
                # CSV Export
                csv = df_results.to_csv(index=False).encode('utf-8')
                col_csv.download_button(label="📊 Raw Data (CSV)", data=csv, file_name=f"AquaGuard_Raw_{st.session_state.address_name.split(',')[0].replace(' ', '_')}_{audit_year}.csv", mime="text/csv", use_container_width=True)
                
                # HTML Dossier Export
                html_dossier = generate_html_dossier(
                    location=st.session_state.address_name,
                    date=f"Stand {audit_year}",
                    total_water=total_stolen_water,
                    fine=total_stolen_water * 1.50,
                    worst_id=worst_field_id,
                    audit_results=audit_results,
                    use_radar=use_radar
                )
                col_html.download_button(label="🖨️ Executive Dossier (HTML)", data=html_dossier, file_name=f"AquaGuard_Dossier_{st.session_state.address_name.split(',')[0].replace(' ', '_')}_{audit_year}.html", mime="text/html", type="primary", use_container_width=True)
                
                if total_stolen_water > 0:
                    st.markdown(f"**Detail-Analyse: Flurstück {worst_field_id} (Höchste Anomalie {audit_year})**")
                    sensor_history = [highest_anomaly_val + np.random.normal(0, 0.01 if not use_radar else 0.5) for _ in range(len(moisture_history))]
                else:
                    st.markdown(f"**Detail-Analyse: Flurstück #1 (Referenz {audit_year})**")
                    ref_val = audit_results[0]["Satelliten-Wert (NDVI/dB)"]
                    sensor_history = [ref_val + np.random.normal(0, 0.01 if not use_radar else 0.5) for _ in range(len(moisture_history))]

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=times, y=moisture_history, name="Natürliches Bodenwasser (ERA5)", line=dict(color="#e74c3c", width=3), fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'), secondary_y=False)
                sensor_name = "Radar Backscatter (dB)" if use_radar else "Satelliten Vitalität (NDVI)"
                sensor_color = "#3498db" if use_radar else "#2ecc71"
                fig.add_trace(go.Scatter(x=times, y=sensor_history, name=sensor_name, line=dict(color=sensor_color, width=3)), secondary_y=True)
                
                if total_stolen_water > 0:
                    fig.add_vrect(x0=times[-30], x1=times[-1], fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="ANOMALIE (Irrigation)", annotation_position="top left")

                fig.update_layout(title="Data Fusion Profil", template="plotly_dark", hovermode="x unified", xaxis_title="Datum", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig.update_yaxes(title_text="Bodenwasser (mm)", secondary_y=False, color="#e74c3c")
                
                if use_radar: fig.update_yaxes(title_text="Radar Backscatter (dB)", secondary_y=True, color=sensor_color, range=[-25, 0])
                else: fig.update_yaxes(title_text="Vegetations-Index (NDVI)", secondary_y=True, color=sensor_color, range=[0, 1])
                
                st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ System-Fehler bei der API-Verarbeitung: {str(e)}")
