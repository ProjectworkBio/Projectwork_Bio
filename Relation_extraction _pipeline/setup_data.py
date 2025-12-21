import os
import requests
import tarfile
import zipfile
import io
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
# Where data_parse.py expects them:
REGULATOME_DIR = os.path.join(DATA_DIR, 'RegulaTome-corpus')
BIORED_DIR = os.path.join(DATA_DIR, 'BioRED')

# URLs
REGULATOME_URL = "https://zenodo.org/records/10808330/files/RegulaTome-corpus.tar.gz?download=1"
BIORED_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"

def download_file(url, desc):
    """Downloads a file with a progress bar."""
    print(f"⬇Downloading {desc}...")
    headers = {'User-Agent': 'Mozilla/5.0'} # Helps avoid 403/429 errors
    
    try:
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        buffer = io.BytesIO()
        
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                buffer.write(data)
                
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error downloading {desc}: {e}")
        return None

def setup_regulatome():
    if os.path.exists(REGULATOME_DIR):
        print(f"RegulaTome directory already exists at: {REGULATOME_DIR}")
        return

    os.makedirs(REGULATOME_DIR, exist_ok=True)
    file_obj = download_file(REGULATOME_URL, "RegulaTome Corpus")
    if file_obj:
        print("Extracting RegulaTome...")
        try:
            with tarfile.open(fileobj=file_obj, mode="r:gz") as tar:
                tar.extractall(path=REGULATOME_DIR)
            print("RegulaTome Extracted.")
        except Exception as e:
            print(f"Error extracting RegulaTome: {e}")

def setup_biored():
    if os.path.exists(BIORED_DIR):
        print(f"BioRED directory already exists at: {BIORED_DIR}")
        return

    os.makedirs(BIORED_DIR, exist_ok=True)
    file_obj = download_file(BIORED_URL, "BioRED Corpus")
    if file_obj:
        print("Extracting BioRED...")
        try:
            with zipfile.ZipFile(file_obj) as z:
                z.extractall(path=BIORED_DIR)
            print("BioRED Extracted.")
            
            # Sanity Check: Ensure the folder is named 'BioRED'
            if not os.path.exists(BIORED_DIR):
                print(f"Warning: Expected folder {BIORED_DIR} not found. Check extracted contents.")
                
        except Exception as e:
            print(f"Error extracting BioRED: {e}")

def setup():
    print("=== STARTING DATA SETUP ===")
    
    setup_regulatome()
    setup_biored()
    
    print("\n=== SETUP COMPLETE ===")
    print(f"RegulaTome path: {REGULATOME_DIR}")
    print(f"BioRED path:     {BIORED_DIR}")

