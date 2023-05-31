import os
import shutil

path = '/ILSVRC2012_img_train_t3'

for root, dirs, files in os.walk(path):
    print(f"Root directory: {root}")
    print(f"Subdirectories: {dirs}")
    print(f"Files: {files}")
    print("Test")
    for file in files:
        if file.endswith(".JPEG"):
            source_path = os.path.join(root, file)
            destination_path = os.path.join(path, file)
            print(f"Moving file: {source_path} to {destination_path}")
            shutil.move(source_path, destination_path)
