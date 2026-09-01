import xml.etree.ElementTree as ET
import re
import zipfile
import io
import time
import requests
import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import cv2

def parse_kml_or_kmz(file_content: bytes, filename: str) -> bytes:
    """
    Decompresses KMZ file to retrieve the KML content, or returns the file
    content directly if it is already a KML.
    """
    if filename.lower().endswith('.kmz'):
        with zipfile.ZipFile(io.BytesIO(file_content)) as z:
            kml_names = [name for name in z.namelist() if name.lower().endswith('.kml')]
            if not kml_names:
                raise ValueError("No KML file found inside the KMZ archive.")
            # Return the first KML file contents
            return z.read(kml_names[0])
    return file_content

def extract_contours_from_kml(kml_content: bytes):
    """
    Parses KML XML content to extract contour lines, elevation values,
    and a list of all raw coordinate points.
    """
    root = ET.fromstring(kml_content)
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    placemarks = root.findall('.//kml:Placemark', namespaces)
    
    contours = []
    all_points = []
    
    for pm in placemarks:
        name_el = pm.find('kml:name', namespaces)
        if name_el is not None and name_el.text:
            try:
                elevation = float(name_el.text.strip())
            except ValueError:
                continue
        else:
            continue
        
        coords_el = pm.find('.//kml:coordinates', namespaces)
        if coords_el is not None and coords_el.text:
            coords_str = coords_el.text.strip()
            pts = []
            for pair in re.split(r'\s+', coords_str):
                if not pair:
                    continue
                parts = pair.split(',')
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        pts.append((lon, lat))
                    except ValueError:
                        pass
            if pts:
                contours.append({
                    'elevation': elevation,
                    'coordinates': pts
                })
                for lon, lat in pts:
                    all_points.append((lon, lat, elevation))
                    
    return contours, all_points

def simplify_contours(contours, max_contours=60, coord_step=5):
    """
    Downsamples the contour dataset to optimize payload size and rendering
    performance in the Leaflet map.
    """
    if not contours:
        return []
    
    elevations = sorted(list(set([c['elevation'] for c in contours])))
    
    if len(elevations) > max_contours:
        step = max(1, len(elevations) // max_contours)
        selected_elevations = set(elevations[::step])
    else:
        selected_elevations = set(elevations)
        
    simplified = []
    for c in contours:
        if c['elevation'] in selected_elevations:
            coords = c['coordinates'][::coord_step]
            if len(c['coordinates']) > 1 and coords[-1] != c['coordinates'][-1]:
                coords.append(c['coordinates'][-1])
            simplified.append({
                'elevation': c['elevation'],
                'coordinates': coords
            })
    return simplified

def fetch_historical_rainfall(lat: float, lon: float, fallback_val: float = 1200.0) -> float:
    """
    Fetches daily rainfall data for the last 10 years from the Open-Meteo API
    and calculates the average annual rainfall in mm.
    """
    start_date = "2016-01-01"
    end_date = "2025-12-31"
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=precipitation_sum&timezone=auto"
    try:
        response = requests.get(url, timeout=0.2)
        if response.status_code == 200:
            data = response.json()
            daily_precip = data.get("daily", {}).get("precipitation_sum", [])
            valid_precip = [p for p in daily_precip if p is not None]
            if valid_precip:
                total_precip = sum(valid_precip)
                avg_annual = total_precip / 10.0
                if avg_annual > 0:
                    return float(round(avg_annual, 2))
    except Exception:
        pass
    return fallback_val

def design_pond(catchment_area_sqm: float, rainfall_mm: float, runoff_coeff: float = 0.4):
    """
    Designs optimal pond dimensions based on catchment runoff and truncated
    pyramid geometry.
    """
    # 1. Total Annual Runoff Volume (m3) = C * R * A
    rainfall_m = rainfall_mm / 1000.0
    annual_runoff_m3 = runoff_coeff * rainfall_m * catchment_area_sqm
    
    # 2. Design Pond Capacity to capture a 50mm storm event runoff, capped at a fraction of annual runoff
    target_capacity_m3 = annual_runoff_m3 * 0.15
    
    # Cap between 150 m3 (minimum viable pond) and 15,000 m3 (large farm pond)
    pond_capacity_m3 = max(150.0, min(target_capacity_m3, 15000.0))
    
    # 3. Geometry calculations: Inverted Truncated Pyramid
    # Standard values: Depth (h) = 3m, Side Slope (z) = 1.5 (stable bank slope)
    h = 3.0
    z = 1.5
    d = 2 * z * h  # 9.0 meters difference between top and bottom sides
    
    # Quadratic Equation to solve for Top Width (W_top):
    # W_top^2 - d * W_top + (d^2 / 3 - V / h) = 0
    a_q = 1.0
    b_q = -d
    c_q = (d ** 2) / 3.0 - (pond_capacity_m3 / h)
    
    discriminant = b_q ** 2 - 4 * a_q * c_q
    
    if discriminant >= 0:
        W_top = (d + np.sqrt(discriminant)) / 2.0
        W_bottom = W_top - d
    else:
        # Fallback if volume is too small for 3m depth and 1.5 side slopes
        W_bottom = 5.0
        W_top = W_bottom + d
        pond_capacity_m3 = (h / 3.0) * (W_top**2 + W_bottom**2 + W_top * W_bottom)
        
    if W_bottom < 2.0:
        # Enforce a minimum bottom width of 3m and recalculate top width
        W_bottom = 3.0
        W_top = W_bottom + d
        pond_capacity_m3 = (h / 3.0) * (W_top**2 + W_bottom**2 + W_top * W_bottom)
        
    return {
        "capacity_m3": float(round(pond_capacity_m3, 2)),
        "capacity_liters": float(round(pond_capacity_m3 * 1000.0, 2)),
        "depth_m": h,
        "side_slope_ratio": z,
        "top_width_m": float(round(W_top, 2)),
        "bottom_width_m": float(round(W_bottom, 2)),
        "excavation_volume_m3": float(round(pond_capacity_m3, 2))
    }

def analyze_contour_map(file_content: bytes, filename: str, runoff_coeff: float = 0.4, custom_rainfall_mm: float = None):
    """
    Performs the full geospatial and hydrological terrain analysis.
    Returns optimal pond location, catchment area, and GeoJSON overlays.
    """
    t_start = time.time()
    
    # 1. Parse KML
    kml_data = parse_kml_or_kmz(file_content, filename)
    contours, all_points = extract_contours_from_kml(kml_data)
    
    if not all_points:
        raise ValueError("No valid coordinates and elevations found in the KML file.")
        
    # Get bounding box and elevation range
    points_arr = np.array(all_points)
    x = points_arr[:, 0]
    y = points_arr[:, 1]
    z = points_arr[:, 2]
    
    lon_min, lon_max = float(x.min()), float(x.max())
    lat_min, lat_max = float(y.min()), float(y.max())
    elev_min_raw, elev_max_raw = float(z.min()), float(z.max())
    
    # Simplify contours for the frontend map layer
    simplified_contours = simplify_contours(contours)
    
    # 2. Build DEM (Digital Elevation Model) Grid
    grid_size = 150
    xi = np.linspace(lon_min, lon_max, grid_size)
    yi = np.linspace(lat_min, lat_max, grid_size)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi)
    
    zi = interpolate.griddata((x, y), z, (xi_mesh, yi_mesh), method='linear')
    nan_mask = np.isnan(zi)
    if np.any(nan_mask):
        zi_nearest = interpolate.griddata((x, y), z, (xi_mesh, yi_mesh), method='nearest')
        zi[nan_mask] = zi_nearest[nan_mask]
        
    # Smooth DEM to filter interpolation noise
    zi_smoothed = ndimage.gaussian_filter(zi, sigma=1.5)
    rows, cols = zi_smoothed.shape
    
    # 3. Flow Routing (D8 Algorithm)
    dr = [-1, 1, 0, 0, -1, -1, 1, 1]
    dc = [0, 0, -1, 1, -1, 1, -1, 1]
    distances = [1.0, 1.0, 1.0, 1.0, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
    
    reverse_flow = { (r, c): [] for r in range(rows) for c in range(cols) }
    sinks = []
    
    for r in range(rows):
        for c in range(cols):
            elev_center = zi_smoothed[r, c]
            max_slope = 0.0
            best_neighbor = None
            
            for idx in range(8):
                nr = r + dr[idx]
                nc = c + dc[idx]
                if 0 <= nr < rows and 0 <= nc < cols:
                    elev_neighbor = zi_smoothed[nr, nc]
                    slope = (elev_center - elev_neighbor) / distances[idx]
                    if slope > max_slope:
                        max_slope = slope
                        best_neighbor = (nr, nc)
                        
            if best_neighbor is not None:
                reverse_flow[best_neighbor].append((r, c))
            else:
                is_boundary = (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)
                sinks.append({
                    'coord': (r, c),
                    'elevation': float(elev_center),
                    'is_boundary': is_boundary
                })
                
    # Calculate cell area in sqm
    lat_center = (lat_min + lat_max) / 2.0
    R_earth = 6378137.0
    dy = ((lat_max - lat_min) / (rows - 1)) * (np.pi / 180.0) * R_earth
    dx = ((lon_max - lon_min) / (cols - 1)) * (np.pi / 180.0) * R_earth * np.cos(np.radians(lat_center))
    cell_area = dx * dy
    
    # 4. Tracing Catchments
    sinks_info = []
    for s in sinks:
        sink_rc = s['coord']
        queue = [sink_rc]
        visited = {sink_rc}
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            for neighbor in reverse_flow[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        catchment_cells = len(visited)
        catchment_area_sqm = catchment_cells * cell_area
        
        sink_lon = float(lon_min + sink_rc[1] * (lon_max - lon_min) / (cols - 1))
        sink_lat = float(lat_min + sink_rc[0] * (lat_max - lat_min) / (rows - 1))
        
        boundary_elevs = []
        for r, c in visited:
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                boundary_elevs.append(float(zi_smoothed[r, c]))
            for idx in range(8):
                nr = r + dr[idx]
                nc = c + dc[idx]
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in visited:
                        boundary_elevs.append(float(zi_smoothed[nr, nc]))
                        
        spill_elevation = min(boundary_elevs) if boundary_elevs else s['elevation']
        sink_depth = spill_elevation - s['elevation']
        
        sinks_info.append({
            'coord': sink_rc,
            'lon': sink_lon,
            'lat': sink_lat,
            'elevation': s['elevation'],
            'is_boundary': s['is_boundary'],
            'catchment_cells': catchment_cells,
            'catchment_area_sqm': catchment_area_sqm,
            'visited_cells': visited,
            'sink_depth': sink_depth
        })
        
    # To avoid placing ponds in the main river/drainage channel, 
    # we exclude the absolute lowest elevations in the terrain (bottom 15% of the elevation range)
    elev_range = elev_max_raw - elev_min_raw
    river_threshold = elev_min_raw + (elev_range * 0.15)
        
    # Prefer interior sinks that have a meaningful depth (e.g., > 0.5m) to avoid riverbed artifacts,
    # and are not located in the main river channel.
    valid_sinks = [s for s in sinks_info if not s['is_boundary'] and s['sink_depth'] >= 0.5 and s['elevation'] > river_threshold]
    
    # Relax depth constraint if no valid sinks found, but keep the river threshold
    if not valid_sinks:
        valid_sinks = [s for s in sinks_info if not s['is_boundary'] and s['sink_depth'] >= 0.2 and s['elevation'] > river_threshold]
        
    candidate_sinks = valid_sinks if valid_sinks else [s for s in sinks_info if not s['is_boundary']]
    
    if not candidate_sinks:
        candidate_sinks = sinks_info
    
    if not candidate_sinks:
        raise ValueError("Could not identify any natural sinks or depressions in the terrain.")
        
    # Sort by catchment area descending
    candidate_sinks.sort(key=lambda x: x['catchment_area_sqm'], reverse=True)
    
    # Take top 5 candidates
    top_n = min(5, len(candidate_sinks))
    top_candidates = candidate_sinks[:top_n]
    
    # Helper: extract boundary polygon for a set of visited cells
    def extract_polygon(visited_cells):
        mask = np.zeros((rows, cols), dtype=np.uint8)
        for r, c in visited_cells:
            mask[r, c] = 255
        contours_cv, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_cv:
            largest_contour = max(contours_cv, key=cv2.contourArea)
            geojson_coords = []
            dlon = (lon_max - lon_min) / (cols - 1)
            dlat = (lat_max - lat_min) / (rows - 1)
            for pt in largest_contour:
                c, r = pt[0][0], pt[0][1]
                lon = float(lon_min + c * dlon)
                lat = float(lat_min + r * dlat)
                geojson_coords.append([lon, lat])
            if geojson_coords:
                geojson_coords.append(geojson_coords[0])
                return {"type": "Polygon", "coordinates": [geojson_coords]}
        return None
    
    # 5. Fetch rainfall once (same region for all candidates)
    center_lat = top_candidates[0]['lat']
    center_lon = top_candidates[0]['lon']
    if custom_rainfall_mm is not None:
        rainfall_mm = custom_rainfall_mm
    else:
        rainfall_mm = fetch_historical_rainfall(center_lat, center_lon)
    
    # 6. Build recommendation list for all top candidates
    pond_sites = []
    for rank, sink in enumerate(top_candidates, start=1):
        polygon = extract_polygon(sink['visited_cells'])
        pd = design_pond(sink['catchment_area_sqm'], rainfall_mm, runoff_coeff)
        pond_sites.append({
            "rank": rank,
            "latitude": float(round(sink['lat'], 6)),
            "longitude": float(round(sink['lon'], 6)),
            "elevation_m": float(round(sink['elevation'], 2)),
            "catchment_area_sqm": float(round(sink['catchment_area_sqm'], 2)),
            "catchment_area_hectares": float(round(sink['catchment_area_sqm'] / 10000.0, 2)),
            "average_annual_rainfall_mm": rainfall_mm,
            "estimated_annual_runoff_m3": float(round(runoff_coeff * (rainfall_mm / 1000.0) * sink['catchment_area_sqm'], 2)),
            "recommended_pond": pd,
            "catchment_polygon": polygon
        })
    
    # The primary recommendation is the #1 ranked site (backward compatible)
    best = pond_sites[0]
    
    processing_time = time.time() - t_start
    
    return {
        "status": "success",
        "processing_time_sec": float(round(processing_time, 3)),
        "pond_recommendation": best,
        "all_pond_sites": pond_sites,
        "contour_summary": {
            "num_contours": len(contours),
            "num_points": len(all_points),
            "bounding_box": {
                "min_lon": float(round(lon_min, 6)),
                "max_lon": float(round(lon_max, 6)),
                "min_lat": float(round(lat_min, 6)),
                "max_lat": float(round(lat_max, 6))
            },
            "elevation_range": {
                "min_m": float(round(elev_min_raw, 2)),
                "max_m": float(round(elev_max_raw, 2))
            }
        },
        "contours_geojson": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": c['coordinates']
                    },
                    "properties": {
                        "elevation": c['elevation']
                    }
                }
                for c in simplified_contours
            ]
        }
    }

