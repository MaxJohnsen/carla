from glob import glob
from pathlib import Path
import re 


def get_best_models(models_path): 
    """
    input: path to folder where different models has been tested 
    return: 
        best_model_paths: a list of paths for each model - where it had the best val loss 
        model_parameter_paths: a list of paths to each model's parameter text 
    """

    best_model_paths = []
    model_parameter_paths = []

    # For each model 
    for model_path in sorted(glob(str(models_path / "*"))):
        min_val_loss = float('inf')
        miv_val_loss_model_path = ""
        # For each file in model folder 
        for model_file_path in glob(str(Path(model_path) / "*.h5")):
            # Get val loss of model
            match = re.search("val(\d+\.*\d*)", model_file_path)
            if match: 
                val_loss = float(match.group(1))
            else:
                print("ERROR: in ", model_path)
                print("Validation loss was not found in the model's file name")
                return None 

            # check if this is min val loss 
            if val_loss <= min_val_loss: 
                min_val_loss = val_loss
                min_val_loss_model_path = Path(model_file_path)

        # Add best model to list 
        best_model_paths.append(min_val_loss_model_path)

        # Get txt-file of model 
        txt_paths = glob(str(Path(model_path) / "*parameters.txt"))
        if len(txt_paths) != 1: 
            print("ERROR in ", model_path)
            print("model folder should have exactly one parameters.txt file")
            return None 
        else: 
            model_parameter_paths.append(Path(txt_paths[0]))

    return best_model_paths, model_parameter_paths
        

def get_parameter_text(path):
    f = open(str(path), "r")
    params = []
    for line in f:
        if 'dataset' not in line and 'epochs' not in line and "batch" not in line:  
            params.append(line.replace("\n", ""))

    return params
    
