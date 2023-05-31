import os
import random
import shutil

def split_images(source_folder, dest_folder):
    # Create destination folders if they don't exist
    train_folder = os.path.join(dest_folder, 'train')
    val_folder = os.path.join(dest_folder, 'val')
    test_folder = os.path.join(dest_folder, 'test')

    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    # Get a list of all image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.JPEG')]

    # Calculate the number of images for each split
    total_images = min(len(image_files), 2000)
    train_count = int(0.7 * total_images)
    val_count = int(0.15 * total_images)
    test_count = total_images - train_count - val_count

    # Randomly shuffle the image files
    random.shuffle(image_files)

    # Move the images to the corresponding folders
    for i in range(total_images):
        image_file = image_files[i]

        if i < train_count:
            dest_path = os.path.join(train_folder, image_file)
        elif i < train_count + val_count:
            dest_path = os.path.join(val_folder, image_file)
        else:
            dest_path = os.path.join(test_folder, image_file)

        src_path = os.path.join(source_folder, image_file)
        shutil.copyfile(src_path, dest_path)

        print(f"Moved {image_file} to {dest_path}")

source_folder = 'ILSVRC2012_img_train_t3'


dest_folder = 'ILSVRC2012_img_train_t3_split'


split_images(source_folder, dest_folder)