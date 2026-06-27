import os
import urllib.request
import zipfile

def download_and_extract():
    url = "https://cricsheet.org/downloads/ipl_csv2.zip"
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    zip_path = os.path.join(raw_dir, "ipl_csv2.zip")
    
    print(f"Downloading IPL data from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
        
    print(f"Extracting zip file to {raw_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        print("Extraction complete.")
        # Remove the zip file to save space
        os.remove(zip_path)
        print("Cleaned up zip file.")
    except Exception as e:
        print(f"Error extracting zip: {e}")

if __name__ == "__main__":
    download_and_extract()
