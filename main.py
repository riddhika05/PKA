import os
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
WORKSPACE_NAME = "MyRAG_Knowledge_Base"
WORKSPACE_DRIVE = "C:"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/setup_workspace")
def setup_workspace(workspace_name=WORKSPACE_NAME, drive=WORKSPACE_DRIVE):
    workspace_path = os.path.join(drive, os.sep, workspace_name)

    if os.path.exists(workspace_path):
        return {"status": "success", "message": f"Workspace already exists at {workspace_path}"}
    
    try:
       
        os.makedirs(workspace_path, exist_ok=True)
        return {"status": "success", "message": f"Workspace created at {workspace_path}"}
    
    except PermissionError:
       
        raise HTTPException(
            status_code=403, 
            detail=f"Permission denied. Could not create folder at: {workspace_path}. Check server user permissions."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred during creation: {str(e)}"
        )
