from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import subprocess
import json
import time
import os

app = FastAPI(title="TruthNet Lite API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    texts: list[str]
    num_processes: int = 1

@app.post("/analyze")
async def analyze_text(req: AnalysisRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(req.texts, f)
        tmp_filename = f.name
    
    # We will use mpiexec or mpirun
    cmd = [
        "mpirun",
        "--allow-run-as-root",
        "-np", str(req.num_processes),
        "python3", "mpi_analyzer.py",
        "--file", tmp_filename
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        
        # Parse the output
        stdout = result.stdout
        # Output contains MPI info and our specific result demarcated by ---MPI_RESULT---
        if "---MPI_RESULT---" in stdout:
            json_str = stdout.split("---MPI_RESULT---")[1].strip()
            mpi_data = json.loads(json_str)
        else:
            raise Exception("Failed to find MPI results in output.")
            
        execution_time = round(end_time - start_time, 4)
        
        return {
            "execution_time_seconds": execution_time,
            "mpi_output": mpi_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

KAGGLE_DF_CACHE = None

@app.get("/analyze_kaggle")
async def analyze_kaggle(limit: int = 1000, np: int = 4):
    global KAGGLE_DF_CACHE
    import pandas as pd
    import os
    
    if KAGGLE_DF_CACHE is None:
        try:
            import kagglehub
            path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
            true_path = os.path.join(path, "True.csv")
            fake_path = os.path.join(path, "Fake.csv")
            
            df_true = pd.read_csv(true_path, encoding='utf-8')
            df_fake = pd.read_csv(fake_path, encoding='utf-8')
            KAGGLE_DF_CACHE = pd.concat([df_true, df_fake]).reset_index(drop=True)
        except Exception as e:
            print("Failed KaggleHub download context, falling back locally:", str(e))
            true_path_loc = os.path.join(os.path.dirname(__file__), 'dataset', 'True.csv')
            fake_path_loc = os.path.join(os.path.dirname(__file__), 'dataset', 'Fake.csv')
            if os.path.exists(true_path_loc) and os.path.exists(fake_path_loc):
                df_true = pd.read_csv(true_path_loc, encoding='utf-8')
                df_fake = pd.read_csv(fake_path_loc, encoding='utf-8')
                KAGGLE_DF_CACHE = pd.concat([df_true, df_fake]).reset_index(drop=True)
            else:
                raise HTTPException(status_code=500, detail="Data evaluation missing globally!")
        
    df = KAGGLE_DF_CACHE.sample(n=min(limit, len(KAGGLE_DF_CACHE))).reset_index(drop=True)
    subset = df['text'].tolist()
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(subset, f)
        tmp_filename = f.name
        
    cmd = [
        "mpirun",
        "--allow-run-as-root",
        "-np", str(np),
        "python3", "mpi_analyzer.py",
        "--file", tmp_filename
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        
        stdout = result.stdout
        if "---MPI_RESULT---" in stdout:
            json_str = stdout.split("---MPI_RESULT---")[1].strip()
            mpi_data = json.loads(json_str)
        else:
            raise Exception("Failed to find MPI results in output.")
            
        execution_time = round(end_time - start_time, 4)
        
        return {
            "execution_time_seconds": execution_time,
            "mpi_output": mpi_data
        }
    except subprocess.CalledProcessError as e:
        print("MPI Error STDERR:", e.stderr)
        raise HTTPException(status_code=500, detail=f"MPI Process Error: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# UNIFIED DEPLOYMENT ROUTING
# Serve React application natively out of backend API 
# =====================================================================
ui_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "truthnet-ui", "dist")

if os.path.exists(ui_path):
    # Mount the standard CSS/JS assets cache explicitly
    if os.path.exists(os.path.join(ui_path, "assets")):
        app.mount("/assets", StaticFiles(directory=os.path.join(ui_path, "assets")), name="assets")

    # Native Catch-All dynamically routes unknown strings naturally ensuring Single-Page App validity
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        file_path = os.path.join(ui_path, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(ui_path, "index.html"))
else:
    print(f"WARNING: React payload cache missing at {ui_path}. Ensure multi-stage Docker build pipeline succeeds.")
