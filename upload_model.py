from huggingface_hub import HfApi

api = HfApi()

print("Creating model repo...")
api.create_repo("chandan1303/smart-city-tinyllama", repo_type="model", exist_ok=True)

print("Uploading trained model files...")
api.upload_folder(
    folder_path="trained_model",
    repo_id="chandan1303/smart-city-tinyllama",
    repo_type="model"
)

print("Done! Model uploaded to https://huggingface.co/chandan1303/smart-city-tinyllama")
