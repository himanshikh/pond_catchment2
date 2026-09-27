import xml.etree.ElementTree as ET
import re
import zipfile
import io
import time
import requests
import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage


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

def fetch_elevation_for_bbox(min_lat: float, min_lon: float, max_lat: float, max_lon: float, grid_dim: int = 25):
    """
    Fetches elevation grid points for a user-selected land area bounding box
    from Open-Meteo elevation API when no KML file is uploaded.
    """
    lats = np.linspace(min_lat, max_lat, grid_dim)
    lons = np.linspace(min_lon, max_lon, grid_dim)
    
    flat_lats = []
    flat_lons = []
    for lat in lats:
        for lon in lons:
            flat_lats.append(round(float(lat), 5))
            flat_lons.append(round(float(lon), 5))
            
    chunk_size = 400
    elevations = []
    for i in range(0, len(flat_lats), chunk_size):
        c_lats = ",".join(map(str, flat_lats[i:i+chunk_size]))
        c_lons = ",".join(map(str, flat_lons[i:i+chunk_size]))
        url = f"https://api.open-meteo.com/v1/elevation?latitude={c_lats}&longitude={c_lons}"
        try:
            resp = requests.get(url, timeout=4.0)
            if resp.status_code == 200:
                elevs = resp.json().get("elevation", [])
                elevations.extend(elevs)
            else:
                elevations.extend([100.0] * len(flat_lats[i:i+chunk_size]))
        except Exception:
            elevations.extend([100.0] * len(flat_lats[i:i+chunk_size]))
            
    all_pts = []
    contours = []
    for idx, (lat, lon) in enumerate(zip(flat_lats, flat_lons)):
        elev = float(elevations[idx]) if idx < len(elevations) and elevations[idx] is not None else 100.0
        all_pts.append((lon, lat, elev))
        
    elev_vals = [pt[2] for pt in all_pts]
    if elev_vals:
        e_min, e_max = min(elev_vals), max(elev_vals)
        levels = np.linspace(e_min, e_max, 10)
        for lvl in levels:
            pts_at_lvl = [pt for pt in all_pts if abs(pt[2] - lvl) < max(0.5, (e_max - e_min)/10.0)]
            if len(pts_at_lvl) >= 2:
                contours.append({
                    'elevation': float(round(lvl, 2)),
                    'coordinates': [(pt[0], pt[1]) for pt in pts_at_lvl]
                })
                
    return contours, all_pts

def analyze_contour_map(file_content: bytes = None, filename: str = None, runoff_coeff: float = 0.4, custom_rainfall_mm: float = None, selected_bbox: list = None):
    """
    Performs full geospatial and hydrological terrain analysis.
    Supports KML/KMZ upload or user-selected land area bounding box on map.
    """
    t_start = time.time()
    
    contours, all_points = [], []

    if file_content and filename:
        kml_data = parse_kml_or_kmz(file_content, filename)
        contours, all_points = extract_contours_from_kml(kml_data)
        
        # If user specified a bounding box filter over uploaded KML
        if selected_bbox and len(selected_bbox) == 4:
            b0, b1, b2, b3 = map(float, selected_bbox)
            min_lat, max_lat = min(b0, b2), max(b0, b2)
            min_lon, max_lon = min(b1, b3), max(b1, b3)

            filtered_pts = [pt for pt in all_points if min_lon <= pt[0] <= max_lon and min_lat <= pt[1] <= max_lat]
            if len(filtered_pts) >= 5:
                all_points = filtered_pts
                filtered_contours = []
                for c in contours:
                    c_pts = [pt for pt in c['coordinates'] if min_lon <= pt[0] <= max_lon and min_lat <= pt[1] <= max_lat]
                    if len(c_pts) >= 2:
                        filtered_contours.append({
                            'elevation': c['elevation'],
                            'coordinates': c_pts
                        })
                contours = filtered_contours
            else:
                raise ValueError(f"No contour lines found inside selected region (Lat: {min_lat:.5f} to {max_lat:.5f}, Lon: {min_lon:.5f} to {max_lon:.5f}). Please draw a box directly over the contour map area.")

    elif selected_bbox and len(selected_bbox) == 4:
        b0, b1, b2, b3 = map(float, selected_bbox)
        min_lat, max_lat = min(b0, b2), max(b0, b2)
        min_lon, max_lon = min(b1, b3), max(b1, b3)
        contours, all_points = fetch_elevation_for_bbox(min_lat, min_lon, max_lat, max_lon)

    else:
        raise ValueError("Please upload a KML/KMZ file or select a land area on the map.")

    if not all_points:
        raise ValueError("No valid terrain coordinate points found for analysis.")

        
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
    
    # Helper: extract boundary polygon for a set of visited cells without OpenCV
    def extract_polygon(visited_cells):
        if not visited_cells:
            return None
        visited_set = set(visited_cells)
        dr = [-1, -1, 0, 1, 1, 1, 0, -1]
        dc = [0, 1, 1, 1, 0, -1, -1, -1]
        
        boundary_cells = []
        for r, c in visited_cells:
            is_boundary = False
            for i in range(8):
                nr, nc = r + dr[i], c + dc[i]
                if (nr, nc) not in visited_set or nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    is_boundary = True
                    break
            if is_boundary:
                boundary_cells.append((r, c))
                
        if not boundary_cells:
            return None
            
        start_cell = min(boundary_cells, key=lambda p: (p[0], p[1]))
        curr = start_cell
        curr_dir = 0
        
        path = [curr]
        max_steps = len(boundary_cells) * 4 + 50
        steps = 0
        
        while steps < max_steps:
            steps += 1
            found = False
            search_start = (curr_dir + 5) % 8
            for i in range(8):
                d = (search_start + i) % 8
                nr, nc = curr[0] + dr[d], curr[1] + dc[d]
                if (nr, nc) in visited_set:
                    curr = (nr, nc)
                    curr_dir = d
                    found = True
                    break
            if not found or curr == start_cell:
                break
            path.append(curr)
            
        if len(path) < 3:
            min_r = min(r for r, c in visited_cells)
            max_r = max(r for r, c in visited_cells)
            min_c = min(c for r, c in visited_cells)
            max_c = max(c for r, c in visited_cells)
            path = [(min_r, min_c), (min_r, max_c), (max_r, max_c), (max_r, min_c)]

        step = max(1, len(path) // 60)
        simplified_path = path[::step]
        if simplified_path[-1] != path[-1]:
            simplified_path.append(path[-1])
            
        dlon = (lon_max - lon_min) / max(1, cols - 1)
        dlat = (lat_max - lat_min) / max(1, rows - 1)
        
        coords = []
        for r, c in simplified_path:
            lon = float(lon_min + c * dlon)
            lat = float(lat_min + r * dlat)
            coords.append([lon, lat])
            
        if coords:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return {"type": "Polygon", "coordinates": [coords]}
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

