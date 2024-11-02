import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import pytz
from scipy.stats import ttest_ind, norm  # Add this line to import the norm distribution
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

my_device = torch.device('cuda' if torch.cuda.is_available() else 'mps' \
                      if torch.backends.mps.is_available() else 'cpu')
print('my_device support :', my_device)

#set params global
num_iterations_all = 100
# num_iterations_all = 5
# pop_size_all = 10
pop_size_all = 50

# set manual, misal ingin dgn cpu
my_device = 'cpu'
print('set_device = ', my_device)
print()

def check_dtype_support(dtype):
    try:
        torch.tensor(1, dtype=dtype, device=my_device)
        # print('sukses bro')
        return True
    except TypeError:
        # print('ada error bro')
        return False
    
def decode_run_params_type1(run_params_tensor):
    # Mendecode sesuai urutan tipe data yang dibutuhkan
    decoded_params = [
        int(run_params_tensor[0].item()),     # int
        float(run_params_tensor[1].item()),   # float
        int(run_params_tensor[2].item()),     # int
        float(run_params_tensor[3].item()),   # float
        float(run_params_tensor[4].item()),   # float
        int(run_params_tensor[5].item())      # int
    ]
    return decoded_params

# Contoh penggunaan
# run_params_tensor = torch.tensor([1.6910, 0.0688, 3.2756, 0.0754, 0.0218, 5.0350])
# decoded_params = decode_run_params_type1(run_params_tensor)

# print("Decoded run_params type 1:", decoded_params)

def decode_run_params_type2(run_params_tensor): 
    decoded_params = [
        int(run_params_tensor[0].item()),     # int
        float(run_params_tensor[1].item()),   # float
        int(run_params_tensor[2].item()),     # int
        float(run_params_tensor[3].item()),   # float
        float(run_params_tensor[4].item()),   # float
        int(run_params_tensor[5].item()),     # int
        "E-MAML" if float(run_params_tensor[-1].item()) <= 0.5 else "E-MAML_Synthetic_E-Reptile"  # kondisi if-else
    ]
    
    return decoded_params

# Contoh penggunaan
# run_params_tensor = torch.tensor([1.6910, 0.0688, 3.2756, 0.0754, 0.0218, 5.0350, 0.5430])
# decoded_params = decode_run_params_type2(run_params_tensor)

# print("Decoded run_params type 2:", decoded_params)

def save_last_model_reptile_checkpoint(model, filename):
    # Dapatkan state_dict dari model
    model_state = model.state_dict()

    # Konversi tensor menjadi list untuk serialisasi JSON
    model_state_serializable = {k: v.numpy().tolist() for k, v in model_state.items()}

    # Simpan model ke file JSON
    with open(filename, 'w') as f:
        json.dump(model_state_serializable, f)

def load_last_model_reptile_checkpoint(model, filename):
    # Muat model dari file JSON
    with open(filename, 'r') as f:
        model_state_serializable = json.load(f)

    # Konversi kembali dari list ke tensor
    model_state = {k: torch.tensor(np.array(v)) for k, v in model_state_serializable.items()}

    # Memuat state_dict ke model
    model.load_state_dict(model_state)
    model.eval()  # Set model ke mode evaluasi
    
def save_last_info_params(
        n_iterations, n_data_all, n_sample, n_train, seed, inner_step_size,
        inner_epochs, outer_stepsize_reptile, outer_stepsize_maml,
        run, final_lossval, filename_last_Model, path_last_Model
    ):
    # Construct the info_params dictionary
    info_params = {
        "n_iterations": n_iterations,
        "n_data_all": n_data_all,
        "n_sample": n_sample,
        "n_train": n_train,
        "seed": seed,
        "inner_step_size": inner_step_size,
        "inner_epochs": inner_epochs,
        "outer_stepsize_reptile": outer_stepsize_reptile,
        "run": run,
        "outer_stepsize_maml": outer_stepsize_maml,
        "final_lossval": float(final_lossval),
        "filename_last_Model": filename_last_Model,
        "path_filename_last_Model": path_last_Model
    }

    # Construct the filename with all the specified information
    filename = (
        f"model_reg_last/model_params_last_{run}_{final_lossval:.3f}_"
        f"{filename_last_Model}.json"
    )

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the info_params dictionary to the file as JSON
    with open(filename, 'w') as json_file:
        json.dump(info_params, json_file, indent=4)

    # print(f"Parameters saved to {filename}")
    # return filename

# Check support for float64
supports_float64 = check_dtype_support(torch.float64)
print(f"Float64 support: {supports_float64}")

# Check support for float32
supports_float32 = check_dtype_support(torch.float32)
print(f"Float32 support: {supports_float32}")