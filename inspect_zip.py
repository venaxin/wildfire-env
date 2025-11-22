import zipfile
import torch
import os
import io

MODEL_FILENAME = "ppo_fire_squad_greedy.zip"

def inspect_model(filepath):
    print(f"--- Inspecting: {filepath} ---")
    
    if not os.path.exists(filepath):
        print("❌ File not found!")
        return

    try:
        with zipfile.ZipFile(filepath, 'r') as archive:
            print("✅ Valid Zip Archive")
            file_list = archive.namelist()
            print(f"📂 Files inside: {file_list}")
            
            # 1. Check for common issue: Nested folders
            # If you see 'ppo_fire_squad_greedy/data' instead of just 'data', that's the problem.
            if not any(f == "data" for f in file_list):
                print("\n⚠️  WARNING: Standard SB3 files not found at root level.")
                print("   Did you re-zip a folder? The files (data, policy.pth) must be at the root of the zip, not inside a subfolder.")
                return

            # 2. Check individual file integrity
            critical_files = ['data', 'policy.pth', 'pytorch_variables.pth']
            for filename in critical_files:
                if filename in file_list:
                    try:
                        with archive.open(filename) as f:
                            content = f.read()
                            # Check for empty or null-filled files
                            if len(content) == 0:
                                print(f"❌ Error: '{filename}' is empty (0 bytes).")
                            elif content.startswith(b'\x00'):
                                print(f"❌ Error: '{filename}' starts with null bytes. File is corrupted.")
                            else:
                                print(f"   File '{filename}' seems okay (Size: {len(content)} bytes).")
                                
                                # Try minimal torch load (if applicable)
                                if filename.endswith('.pth'):
                                    try:
                                        torch.load(io.BytesIO(content), map_location='cpu')
                                        print(f"   ✅ torch.load('{filename}') successful.")
                                    except Exception as e:
                                        print(f"   ❌ torch.load('{filename}') FAILED: {e}")
                    except Exception as e:
                        print(f"   ❌ Failed to read '{filename}': {e}")
                else:
                    print(f"⚠️  Warning: Critical file '{filename}' missing from archive.")

    except zipfile.BadZipFile:
        print("❌ Error: The file is not a valid zip archive.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    inspect_model(MODEL_FILENAME)