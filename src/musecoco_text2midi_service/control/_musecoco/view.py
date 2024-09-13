import os
import sys
import shutil
from .text2attribute_model import Text2AttributePredictor, prepare_data
from .attribute2music_model.linear_mask import interactive_dict_v5_1billion

def init_text2attribute():
    # Call the main function to process simulated inputs
    predictor = Text2AttributePredictor()
    
    return predictor

def prepare_stage2(source_path, destination_path):
    # Step 2: Prepare intermediate data by executing necessary scripts
    # Move to the directory for text2attribute model processing
    # Run `stage2_pre.py` - you mentioned it's a script that can be imported
    prepare_data()

    # Move generated `infer_test.bin` to the appropriate directory
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(source_path, destination_path)
    
def init_attribute2midi():
    # Step 9: Call cli_main with modified arguments
    interactive_dict_v5_1billion.seed_everything(2024)  # Set random seed
    
    return interactive_dict_v5_1billion.Attribute2MusicPredictor()
