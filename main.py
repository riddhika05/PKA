import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# CONFIG
WORKSPACE_NAME = "MyRAG_Knowledge_Base"
WORKSPACE_DRIVE = "C:"

# CREATE APP
app = FastAPI()

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENDPOINTS

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
            detail=f"Permission denied. Could not create folder at: {workspace_path}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    
    try:
        workspace_path = os.path.join(WORKSPACE_DRIVE, os.sep, WORKSPACE_NAME)
        
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            if file.filename:
                # Preserve folder structure from webkitRelativePath
                # file.filename will be like "FolderName/subfolder/file.txt"
                safe_filename = file.filename.replace(" ", "_")
                file_path = os.path.join(workspace_path, safe_filename)
                
                # CRITICAL: Create parent directories BEFORE trying to write file
                parent_dir = os.path.dirname(file_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                
                # Write the file
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                uploaded_files.append({
                    "filename": file.filename,
                    "saved_as": safe_filename,
                    "size": len(content),
                    "path": file_path
                })
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(uploaded_files)} file(s)",
            "uploaded_files": uploaded_files,
            "workspace_path": workspace_path
        }
        
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Permission denied. Could not write to workspace directory."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload error: {str(e)}"
        )