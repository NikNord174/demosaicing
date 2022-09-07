import os
import PIL
import rawpy


FOLDER_PATH = '/Users/nikolai/CV/demosaicing/intern_task/'


def convert_data():
    for infile in os.listdir(FOLDER_PATH):
        file_path = FOLDER_PATH + infile
        raw = rawpy.imread(file_path)
        rgb = raw.postprocess(use_camera_wb=True)
        PIL.Image.fromarray(rgb).save(f'{infile}.png')
        raw.close()
