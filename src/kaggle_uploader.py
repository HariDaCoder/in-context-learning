import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile


def zip_directory(src_dir, zip_path):
    print(f"[INFO] Zipping directory {src_dir} to {zip_path}...")
    shutil.make_archive(zip_path.replace(".zip", ""), "zip", src_dir)
    print("[INFO] Zip completed.")


def zip_final_models_only(run_root, zip_path):
    print(f"[INFO] Scanning {run_root} for final model files...")
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(run_root):
            # Check if this directory looks like a run dir (contains config.yaml and state.pt)
            if "config.yaml" in files and "state.pt" in files:
                run_name = os.path.basename(root)
                print(f"[INFO] Adding run: {run_name}")
                
                # Add final state
                state_path = os.path.join(root, "state.pt")
                zipf.write(state_path, arcname=os.path.join(run_name, "state.pt"))
                
                # Add config
                config_path = os.path.join(root, "config.yaml")
                zipf.write(config_path, arcname=os.path.join(run_name, "config.yaml"))
                
                # Add train_losses if it exists
                losses_path = os.path.join(root, "train_losses.json")
                if os.path.exists(losses_path):
                    zipf.write(losses_path, arcname=os.path.join(run_name, "train_losses.json"))
                    
                count += 1
                
    print(f"[INFO] Added {count} final model runs to {zip_path}.")


def upload_to_bashupload(file_path):
    file_name = os.path.basename(file_path)
    url = f"https://bashupload.com/{file_name}"
    print(f"[INFO] Uploading {file_path} to {url}...")
    
    # Run curl to upload
    cmd = ["curl", "-T", file_path, url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("[SUCCESS] Upload finished!")
        print("\n" + "=" * 60)
        print(f"DOWNLOAD LINK:\n{result.stdout.strip()}")
        print("=" * 60 + "\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to upload via curl: {e.stderr}")
        # Try a fallback to transfer.sh if curl fails
        print("[INFO] Trying fallback to transfer.sh...")
        fallback_cmd = ["curl", "--upload-file", file_path, f"https://transfer.sh/{file_name}"]
        try:
            fb_res = subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
            print("[SUCCESS] Upload finished via fallback!")
            print("\n" + "=" * 60)
            print(f"DOWNLOAD LINK:\n{fb_res.stdout.strip()}")
            print("=" * 60 + "\n")
        except Exception as err:
            print(f"[ERROR] Fallback also failed: {err}")


def main():
    parser = argparse.ArgumentParser(description="Kaggle File Zipper and Uploader Helper.")
    parser.add_argument("--dir", type=str, default=None, help="Directory to zip and upload.")
    parser.add_argument(
        "--run_root", 
        type=str, 
        default=None, 
        help="Root directory of runs. Used with --only_final to download only state.pt + configs."
    )
    parser.add_argument(
        "--only_final", 
        action="store_true", 
        help="If set, only state.pt, config.yaml, and train_losses.json are zipped from --run_root."
    )
    parser.add_argument("--file", type=str, default=None, help="Single file to upload without zipping.")
    parser.add_argument("--name", type=str, required=True, help="Name of the zipped file / upload package (e.g. results.zip).")
    
    args = parser.parse_args()

    if not args.name.endswith(".zip") and (args.dir or args.run_root):
        args.name += ".zip"

    temp_dir = tempfile.gettempdir()
    zip_out_path = os.path.join(temp_dir, args.name)

    # Remove existing zip if any
    if os.path.exists(zip_out_path):
        os.remove(zip_out_path)

    if args.file:
        upload_to_bashupload(args.file)
    elif args.run_root and args.only_final:
        zip_final_models_only(args.run_root, zip_out_path)
        upload_to_bashupload(zip_out_path)
        # Clean up local zip
        if os.path.exists(zip_out_path):
            os.remove(zip_out_path)
    elif args.dir:
        zip_directory(args.dir, zip_out_path)
        upload_to_bashupload(zip_out_path)
        # Clean up local zip
        if os.path.exists(zip_out_path):
            os.remove(zip_out_path)
    else:
        print("[ERROR] Please provide --dir, --run_root with --only_final, or --file.")


if __name__ == "__main__":
    main()
