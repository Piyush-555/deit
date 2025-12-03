import os
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
from torchvision.datasets.folder import default_loader # Requires torchvision
import time
import warnings
from contextlib import ExitStack

# --- Configuration ---
# Set the root of your ImageNet validation data
DATA_ROOT = "/mnt/data/Public_datasets/ImageNet21K/winter21_whole/" 
# Number of processes to run concurrently
NUM_PROCESSES = mp.cpu_count() 
# List of image extensions to check
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
# Output file name for the list of corrupt files
CORRUPT_LOG_FILE = "corrupt_imagenet_files.txt"

# --- Worker Function ---
def check_file_corruption(file_path: Path) -> Tuple[bool, str]:
    """
    Attempts to load a single image file, treating specific PIL UserWarnings as errors.
    Returns (is_corrupt, path).
    """
    path_str = str(file_path)

    # 1. Extension check
    if not path_str.lower().endswith(IMAGE_EXTENSIONS):
        return (False, path_str) 

    # 2. Loadability and Warning Check
    try:
        with ExitStack() as stack:
            # Set up the context to treat specific warnings as exceptions
            stack.enter_context(warnings.catch_warnings(record=True))
            
            # --- Promote PIL warnings to exceptions ---
            # 1. Corrupt EXIF data (often from TiffImagePlugin)
            warnings.filterwarnings(
                "error", 
                category=UserWarning, 
                message="Corrupt EXIF data.*"
            )
            # 2. Truncated File Read (often from ImageFile.load_end)
            warnings.filterwarnings(
                "error", 
                category=UserWarning, 
                message="Truncated File Read.*"
            )

            # Attempt to load the image file
            default_loader(path_str)
            return (False, path_str) # Not corrupt
            
    except Exception:
        # Catches standard loading errors AND the promoted warnings
        return (True, path_str) # Corrupt

# --- Main Script ---
def find_corrupt_files_parallel(root_dir: str, num_processes: int, log_file: str):
    """
    Scans the root directory using a multiprocessing pool to find corrupted files.
    """
    print(f"Starting parallel scan using {num_processes} processes...")
    start_time = time.time()
    
    # 1. Collect all file paths
    all_files: List[Path] = []
    root_path = Path(root_dir)
    if not root_path.is_dir():
        print(f"Error: Root directory not found at {root_dir}")
        return

    # Recursively find all files in the root directory
    for entry in root_path.rglob('*'):
        if entry.is_file():
            all_files.append(entry)

    total_files = len(all_files)
    print(f"Found {total_files:,} files to check.")

    if not all_files:
        print("No files found. Exiting.")
        return

    corrupt_files: List[str] = []
    
    # 2. Use Multiprocessing Pool
    with mp.Pool(processes=num_processes) as pool:
        # Map the checker function to all file paths
        results = pool.imap(check_file_corruption, all_files, chunksize=100)
        
        for i, (is_corrupt, path) in enumerate(results):
            if is_corrupt:
                corrupt_files.append(path)
                
            # Print progress update
            if (i + 1) % 10000 == 0 or (i + 1) == total_files:
                elapsed = time.time() - start_time
                print(f"Processed {i + 1:,}/{total_files:,} files. Found {len(corrupt_files)} corrupt files. Time: {elapsed:.2f}s")
                
    # 3. Log Results
    if corrupt_files:
        with open(log_file, 'w') as f:
            f.write('\n'.join(corrupt_files) + '\n')
        print(f"\n✅ Scan complete! Found {len(corrupt_files)} corrupted files.")
        print(f"📝 Corrupted file paths saved to: {log_file}")
    else:
        print("\n✅ Scan complete! No corrupted image files found.")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    find_corrupt_files_parallel(DATA_ROOT, NUM_PROCESSES, CORRUPT_LOG_FILE)