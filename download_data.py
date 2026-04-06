from roboflow import Roboflow
import os

def download_helmet_dataset():
    """
    Downloads the 'Construction Site Safety' dataset from Roboflow in YOLOv8 format.
    Uses the user's API Key provided in the instructions.
    """
    # Initialize Roboflow with the Private API Key
    # Note: User provided 1Y562pbo8vu07f3ZGQyB in their screenshot
    rf = Roboflow(api_key="1Y562pbo8vuO7f3ZGQyB")

    # Define the project and version
    # 'construction-site-safety' is a high-quality dataset
    project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
    version = project.version(24) # Version 24 is a stable YOLOv8 format release

    # Create the datasets directory if it doesn't exist
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    # Download the dataset in YOLOv8 format
    print("Downloading dataset from Roboflow Universe...")
    dataset = version.download("yolov8", location="datasets")
    
    print(f"Dataset downloaded successfully to: {dataset.location}")
    print("Classes in this dataset:", dataset.classes)

if __name__ == "__main__":
    download_helmet_dataset()
