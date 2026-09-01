from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from app.analyzer import analyze_contour_map

app = FastAPI(
    title="Village Pond Planning & Catchment Analysis Backend API",
    description="Automated backend API for continuous terrain elevation modeling, optimal village pond location ranking, and exact hydrological catchment delineation from KML/KMZ contour maps.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

@app.post("/api/analyze-contour")
async def analyze_contour(
    contour_map: UploadFile = File(None),
    file: UploadFile = File(None),
    runoff_coefficient: float = Form(0.4),
    rainfall_mm: float = Form(None)
):
    """
    Accepts a KML or KMZ contour map file and runs a terrain analysis to find
    the optimal pond location and its catchment area.
    """
    upload_file = contour_map or file
    if not upload_file:
        raise HTTPException(
            status_code=400,
            detail="Missing file. Please upload a KML/KMZ file under variable name 'contour_map' or 'file'."
        )
        
    filename = upload_file.filename
    if not (filename.lower().endswith('.kml') or filename.lower().endswith('.kmz')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only KML (.kml) and KMZ (.kmz) files are supported."
        )
        
    try:
        # Read uploaded file content
        file_content = await upload_file.read()
        
        # Perform terrain and catchment analysis
        result = analyze_contour_map(
            file_content=file_content,
            filename=filename,
            runoff_coeff=runoff_coefficient,
            custom_rainfall_mm=rainfall_mm
        )
        return result
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during terrain analysis: {str(e)}"
        )

# Backward compatibility routes
@app.post("/analyzeContour")
async def analyze_contour_legacy(
    contour_map: UploadFile = File(None),
    file: UploadFile = File(None),
    runoff_coefficient: float = Form(0.4),
    rainfall_mm: float = Form(None)
):
    return await analyze_contour(contour_map, file, runoff_coefficient, rainfall_mm)

@app.post("/findCatchment")
async def find_catchment_legacy(
    contour_map: UploadFile = File(None),
    file: UploadFile = File(None),
    runoff_coefficient: float = Form(0.4),
    rainfall_mm: float = Form(None)
):
    return await analyze_contour(contour_map, file, runoff_coefficient, rainfall_mm)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """
    Serves the interactive, premium web interface for the catchment analysis.
    """
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard template not found.")
        
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content
