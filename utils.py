import os
import random
import shutil

"""
# Common_utils
"""

def save_model(model, model_name="model"):
    model.save(f'{model_name}.h5')
    print(f"{model_name}.h5 saved.")


def moving_files_to_dir(path, dest_path, split_coef=0.3):
    moved_files =[]
    files = os.listdir(path)
    moved_files_number = round(len(files)*split_coef)
    
    for i in range(moved_files_number):
        index = random.randrange(0, len(files))
        moved_files.append(files.pop(index))

    for f in moved_files:
        shutil.move(f"{path}/{f}", f"{dest_path}/{f}")
