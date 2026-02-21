import os
import shutil

# Source and target paths
SOURCE_PATH = "dataset/medical_data"
TARGET_PATH = "dataset"

for organ in os.listdir(SOURCE_PATH):
    organ_path = os.path.join(SOURCE_PATH, organ)

    if os.path.isdir(organ_path):
        target_organ_path = os.path.join(TARGET_PATH, organ)
        os.makedirs(target_organ_path, exist_ok=True)

        for disease in os.listdir(organ_path):
            disease_path = os.path.join(organ_path, disease)

            if os.path.isdir(disease_path):
                for img in os.listdir(disease_path):
                    src_img = os.path.join(disease_path, img)

                    # rename to avoid overwrite
                    new_name = disease + "_" + img
                    dst_img = os.path.join(target_organ_path, new_name)

                    shutil.copy(src_img, dst_img)

print("Dataset flattened successfully!")