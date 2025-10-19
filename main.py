import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tkinter
from typing import List
WORKSPACE_NAME = "MyRAG_Knowledge_Base"
WORKSPACE_DRIVE = "C:"
app = FastAPI()
import tkinter as tk
from tkinter import filedialog, messagebox
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

@app.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files to the workspace
    """
    try:
        # Get the workspace path
        workspace_path = os.path.join(WORKSPACE_DRIVE, os.sep, WORKSPACE_NAME)
        
        # Create workspace if it doesn't exist
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            if file.filename:
                # Create a safe filename
                safe_filename = file.filename.replace(" ", "_")
                file_path = os.path.join(workspace_path, safe_filename)
                
                # Save the file
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                uploaded_files.append({
                    "filename": file.filename,
                    "saved_as": safe_filename,
                    "size": len(content)
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
            detail=f"An error occurred during file upload: {str(e)}"
        )
