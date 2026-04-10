import os, glob, zipfile, requests, io, time, shutil
import numpy as np
import cdsapi
import netCDF4 as nc
import rasterio
from rasterio.warp import transform_bounds
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter, label
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()
USER = os.getenv('COPERNICUS_USER')
PASSWORD = os.getenv('COPERNICUS_PW')
CDS_KEY = os.getenv('CDS_API_KEY')

# ─────────────────────────────────────────────
# SENTINEL-2 (NDVI)
# ─────────────────────────────────────────────

def get_sentinel_token():
    """OAuth2-Token vom Copernicus Dataspace – mit Auto-Retry."""
    for attempt in range(3):
        try:
            r = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data={'client_id': 'cdse-public', 'grant_type': 'password', 'username': USER, 'password': PASSWORD},
                timeout=15
            )
            r.raise_for_status()
            return r.json()['access_token']
        except Exception as e:
            if attempt == 2:
                raise ConnectionError(f"Token-Abruf fehlgeschlagen: {e}")
            time.sleep(2)

def search_sentinel2(lat, lon, date_start, date_end, token, cloud_cover=60):
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    point = f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})')"
    params = {
        "$filter": (f"{point} and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and "
                    f"att/OData.CSC.DoubleAttribute/Value lt {cloud_cover}.0) and "
                    f"ContentDate/Start gt {date_start} and ContentDate/Start lt {date_end} and contains(Name,'S2A_MSIL2A')"),
        "$top": 1, "$orderby": "ContentDate/Start desc"
    }
    # HIER ist der 60-Sekunden Timeout fest eingebaut
    r = requests.get(url, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    results = r.json().get('value', [])
    return results[0] if results else None

def download_sentinel2(product, token, label_text=""):
    name = product['Name']
    praefix = name.rsplit('_', 1)[0]
    cached = glob.glob(f"{praefix}*.SAFE")
    if cached:
        print(f"   ✓ Bereits vorhanden: {cached[0]}")
        return cached[0]

    print(f"   📥 Lade {label_text} herunter (Dies dauert kurz, danach wird aufgeräumt)...")
    prod_id = product['Id']
    d_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({prod_id})/$value"

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    r = session.get(d_url, allow_redirects=False, timeout=30)
    while r.status_code in (301, 302, 303, 307):
        d_url = r.headers['Location']
        r = session.get(d_url, allow_redirects=False, timeout=30)

    zip_path = f"{name}.zip"
    with session.get(d_url, stream=True, timeout=120) as res:
        res.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in res.iter_content(8192):
                f.write(chunk)

    print(f"   📦 Entpacke Archiv...")
    zipfile.ZipFile(zip_path).extractall(".")
    os.remove(zip_path)
    return glob.glob(f"{praefix}*.SAFE")[0]

def calc_ndvi_from_safe(safe_dir):
    b4_files = glob.glob(f"{safe_dir}/**/*_B04_10m.jp2", recursive=True)
    b8_files = glob.glob(f"{safe_dir}/**/*_B08_10m.jp2", recursive=True)
    if not b4_files or not b8_files:
        raise FileNotFoundError(f"B04/B08 Bänder nicht gefunden in {safe_dir}")

    with rasterio.open(b4_files[0]) as ds_red:
        red = ds_red.read(1).astype('float32')
        bounds = ds_red.bounds
        crs = ds_red.crs
    with rasterio.open(b8_files[0]) as ds_nir:
        nir = ds_nir.read(1).astype('float32')
    
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi, bounds, crs

def extract_ndvi_for_location(lat, lon, ndvi, bounds, crs, radius_px=80):
    west, south, east, north = transform_bounds(crs, 'EPSG:4326', *bounds)
    h, w = ndvi.shape
    px_col = int(((lon - west) / (east - west)) * w)
    px_row = int(((north - lat) / (north - south)) * h)
    r_s, r_e = max(0, px_row - radius_px), min(h, px_row + radius_px)
    c_s, c_e = max(0, px_col - radius_px), min(w, px_col + radius_px)
    crop = ndvi[r_s:r_e, c_s:c_e]
    
    map_bounds = [[north - (r_e/h)*(north-south), west + (c_s/w)*(east-west)],
                  [north - (r_s/h)*(north-south), west + (c_e/w)*(east-west)]]

    threshold = 0.35 if (crop > 0.35).mean() > 0.05 else 0.30
    mask = crop > threshold
    mask = binary_opening(mask, structure=np.ones((5, 5)), iterations=2)
    mask = binary_closing(mask, structure=np.ones((7, 7)), iterations=1)
    labeled, n_features = label(mask)
    for i in range(1, n_features + 1):
        if np.sum(labeled == i) < 100:
            mask[labeled == i] = False

    stats = {}
    if mask.any():
        stats['mean_ndvi'] = float(np.nanmean(crop[mask]))
        stats['summer_ndvi'] = float(np.nanmean(crop[mask])) 
        stats['vmin'] = float(np.nanpercentile(crop[mask], 5))
        stats['vmax'] = float(np.nanpercentile(crop[mask], 95))
        stats['vegetation_coverage'] = float(mask.mean())
    else:
        stats = {'mean_ndvi': 0.2, 'summer_ndvi': 0.2, 'vmin': 0.1, 'vmax': 0.5, 'vegetation_coverage': 0.0}
    return crop, mask, map_bounds, stats

def get_ndvi_for_site(lat, lon, year):
    print(f"\n🛰️  Prüfe Daten für ({lat:.4f}, {lon:.4f}), Jahr {year}...")
    
    cache_file = f"ndvi_microcache_{lat:.4f}_{lon:.4f}_{year}.npz"
    if os.path.exists(cache_file):
        print(f"   ✓ Winzigen Micro-Cache gefunden. Lade in Millisekunden...")
        data = np.load(cache_file, allow_pickle=True)
        return data['result'].item()

    try:
        token = get_sentinel_token()
        start = f"{year}-05-01T00:00:00.000Z"
        end = f"{year}-09-30T23:59:59.000Z"
        product = search_sentinel2(lat, lon, start, end, token)
        
        if not product:
            print("   ⚠️  Kein Produkt gefunden (Fallback auf weichere Wolkengrenze).")
            product = search_sentinel2(lat, lon, f"{year}-01-01T00:00:00.000Z", f"{year}-12-31T23:59:59.000Z", token, cloud_cover=80)
            if not product:
                return None

        safe_dir = download_sentinel2(product, token, f"Sentinel-2 {year}")
        ndvi, bounds, crs = calc_ndvi_from_safe(safe_dir)
        crop, mask, map_bounds, stats = extract_ndvi_for_location(lat, lon, ndvi, bounds, crs)

        result = {'ndvi_crop': crop, 'ndvi_mask': mask, 'map_bounds': map_bounds, 'stats': stats, 'source': 'sentinel2_real'}
        
        np.savez(cache_file, result=result)
        
        print(f"   🧹 Räume auf: Lösche riesigen Satelliten-Ordner um Festplatte zu schonen...")
        shutil.rmtree(safe_dir, ignore_errors=True)

        print(f"   ✓ NDVI extrahiert und gespeichert: Mean={stats['mean_ndvi']:.3f}")
        return result

    except Exception as e:
        print(f"   ⚠️  Sentinel-2 Fehler: {e}")
        return None

# ─────────────────────────────────────────────
# ERA5 BODENFEUCHTE (CDS API)
# ─────────────────────────────────────────────

def get_era5_moisture(lat, lon, year):
    print(f"\n🌍 Lade ERA5 Bodenfeuchtedaten für Jahr {year}...")
    cache_file = f"era5_moisture_{lat:.2f}_{lon:.2f}_{year}.npy"
    if os.path.exists(cache_file):
        print(f"   ✓ Cache geladen.")
        return np.load(cache_file)

    try:
        if not CDS_KEY: raise ValueError("CDS_API_KEY nicht gesetzt in .env")
        nc_file = f"era5_raw_{year}.nc"
        area = [lat + 1, lon - 1, lat - 1, lon + 1]

        # Korrekter API-Link
        c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=CDS_KEY, quiet=True)
        c.retrieve('reanalysis-era5-land', {
            'variable': 'volumetric_soil_water_layer_1', 'year': str(year),
            'month': [f'{m:02d}' for m in range(1, 13)], 'day': [f'{d:02d}' for d in range(1, 32)],
            'time': '12:00', 'area': area, 'format': 'netcdf'
        }, nc_file)

        ds = nc.Dataset(nc_file)
        swvl1 = ds.variables['swvl1'][:] 
        daily_values = np.array([float(np.nanmean(swvl1[t])) for t in range(swvl1.shape[0])])
        moisture_mm = daily_values * 70 

        if len(moisture_mm) > 365: moisture_mm = moisture_mm[:365]
        elif len(moisture_mm) < 365: moisture_mm = np.pad(moisture_mm, (0, 365 - len(moisture_mm)), mode='edge')

        np.save(cache_file, moisture_mm)
        os.remove(nc_file)
        print(f"   ✓ ERA5 extrahiert und gespeichert.")
        return moisture_mm
    except Exception as e:
        print(f"   ⚠️  ERA5 Fehler: {e}")
        return None

# ─────────────────────────────────────────────
# ANOMALIE-ERKENNUNG (WISSENSCHAFTLICH)
# ─────────────────────────────────────────────

def detect_irrigation_anomaly(ndvi_summer, moisture_series):
    if moisture_series is not None:
        summer_moisture = float(np.nanmean(moisture_series[150:270]))
        drought_condition = summer_moisture < 25.0
    else:
        drought_condition, summer_moisture = True, 18.0

    high_ndvi = ndvi_summer > 0.55
    is_anomaly = drought_condition and high_ndvi

    confidence = 0.0
    if is_anomaly:
        ndvi_factor = min((ndvi_summer - 0.55) / 0.3, 1.0)
        drought_factor = min((25.0 - summer_moisture) / 25.0, 1.0) if moisture_series is not None else 0.7
        confidence = (ndvi_factor * 0.6 + drought_factor * 0.4) * 100

    return {'is_anomaly': bool(is_anomaly), 'summer_moisture_mm': float(summer_moisture) if moisture_series is not None else None,
            'summer_ndvi': float(ndvi_summer), 'drought_condition': bool(drought_condition), 'confidence_pct': round(confidence, 1)}

def estimate_illegal_water_volume(area_ha, ndvi_current, moisture_mm, crop_type='Tomaten', licensed_m3=0.0):
    crop_water_needs = {'Tomaten': 5000, 'Alfalfa': 12000, 'Zitrusfrüchte': 7500, 'Mandeln': 8500, 'Trauben': 6000, 'Soja': 5500, 'Oliven': 4500, 'Beeren': 4000, 'Baumwolle': 11000, 'Weizen': 3500}
    water_need = crop_water_needs.get(crop_type, 6000)

    if moisture_mm is not None:
        summer_rain_mm = float(np.nanmean(moisture_mm[150:270]))
        rain_contribution = min(summer_rain_mm * 8, water_need * 0.6)
    else:
        rain_contribution = water_need * 0.2 
        
    total_irrigation = max(water_need - rain_contribution, 0) * area_ha
    illegal = max(total_irrigation - licensed_m3, 0)
    return round(illegal, 0)
