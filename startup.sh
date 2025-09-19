#!/bin/bash

# --- Configuration ---
# The target directory inside the container, which is mounted to the persistent disk.
TARGET_DIR="/app/product_db"
# A file to check for existence to see if the DB is already extracted.
CHECK_FILE="${TARGET_DIR}/chroma.sqlite3"
# The ID of your Google Drive Folder.
GDRIVE_FOLDER_ID="1-tktqeXhjjvpACRWHd9xE2UgDzpt4aco"
ZIP_FILE_NAME="product_db.zip"

# --- Main Logic ---
echo "--- Running Application Startup Script ---"

# Check if the database directory is already populated
if [ -f "$CHECK_FILE" ]; then
    echo "✅ Database found at ${TARGET_DIR}. Skipping download."
else
    echo "🟡 Database not found. Starting download from Google Drive..."
    
    # Ensure the target directory exists and navigate into it
    mkdir -p ${TARGET_DIR}
    cd ${TARGET_DIR}
    
    # Download the folder from Google Drive as a zip file
    echo "Downloading folder with ID: ${GDRIVE_FOLDER_ID}"
    gdown --folder "${GDRIVE_FOLDER_ID}" -O "${ZIP_FILE_NAME}" --quiet
    
    if [ $? -eq 0 ]; then
        echo "✅ Download complete. Extracting files..."
        
        # Unzip the contents into the current directory
        unzip -q "${ZIP_FILE_NAME}"
        
        if [ $? -eq 0 ]; then
            echo "✅ Extraction successful."
            # Clean up the zip file to save space
            rm "${ZIP_FILE_NAME}"
        else
            echo "❌ ERROR: Failed to extract ${ZIP_FILE_NAME}."
            exit 1
        fi
    else
        echo "❌ ERROR: Failed to download from Google Drive."
        exit 1
    fi
    
    # Go back to the app root directory
    cd /app
fi

# --- Start the Application ---
echo "--- Starting Uvicorn Server ---"
# 'exec' replaces the script process with the uvicorn process, which is the correct way to run.
exec uvicorn main:app --host 0.0.0.0 --port 8000
