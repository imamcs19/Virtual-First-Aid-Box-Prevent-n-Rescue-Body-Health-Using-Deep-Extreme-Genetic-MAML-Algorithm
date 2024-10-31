## n_hidden = k (dinamis) untuk Tiny dataset
## 
## Model 1 utilize data (Migraine + Vertigo + Sariawan (di mulut/lidah))
## Model 2 utilize data (Sakit Gigi + Radang Tenggorokan)
## Model 3  As Baseline or Grouth Truth utilize all data dengan n_hidden = k (dinamis), dimana k = 6 (hidden_layers = [100, 50, 25, 12, 6, 3])

import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime

# set var global
n_input = 14
# n_hidden1 = 100
# n_hidden2 = 50
# hidden_layers = 
hidden_layers = [100, 50, 25, 12, 6, 3] 
n_output = 44


# Define ELM Model with dynamic hidden layers for regression
class ELMRegression(nn.Module):
    def __init__(self, n_input, hidden_layers, n_output):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_layer_size = n_input

        # Dynamically create hidden layers
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_layer_size, hidden_size))
            prev_layer_size = hidden_size

        # Output layer
        self.output = nn.Linear(prev_layer_size, n_output)

    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))  # Sigmoid activation for each hidden layer
        return self.output(x)  # Linear output layer for regression

# Function to save model parameters in JSON format
def save_model_json(model, file_path):
    model_params = {
        f"layer_{i}": layer.weight.detach().numpy().tolist()
        for i, layer in enumerate(model.layers)
    }
    model_params.update({
        f"layer_{i}_bias": layer.bias.detach().numpy().tolist()
        for i, layer in enumerate(model.layers)
    })
    model_params["output_weights"] = model.output.weight.detach().numpy().tolist()
    model_params["output_bias"] = model.output.bias.detach().numpy().tolist()

    with open(file_path, 'w') as json_file:
        json.dump(model_params, json_file)
    print(f"Model saved as JSON at {file_path}")
    
# Fungsi untuk mencatat dan menyimpan loss dalam format JSON berdasarkan epoch
def save_loss_json(loss_per_epoch, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(loss_per_epoch, json_file)
    print(f"Loss per epoch saved as JSON at {file_path}")

# Fungsi untuk mem-plot dan menyimpan hasil training (loss per epoch)
def plot_loss(loss_per_epoch, sheet_name, name_unik):
    epochs = list(range(1, len(loss_per_epoch) + 1))
    plt.figure()
    plt.plot(epochs, loss_per_epoch, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss per Epoch for {sheet_name}')
    plt.legend()

    # Menyimpan plot dalam format PNG dan PDF
    plt.savefig(f'loss/loss_plot_{sheet_name}_{name_unik}.png')
    plt.savefig(f'loss/loss_plot_{sheet_name}_{name_unik}.pdf')
    print(f"Loss plot saved as PNG and PDF for {sheet_name}")

    plt.close()  # Menutup plot setelah selesai
    
# Function to plot loss for all sheets in a single file
# def plot_loss_all(loss_per_epoch_dict, sheets, gen_name_unik):
#     plt.figure(figsize=(10, 6))  # Set a larger figure size for better readability
    
#     # Plot the loss for each sheet
#     for sheet_name in sheets:
#         plt.plot(loss_per_epoch_dict[sheet_name], label=f'Loss for {sheet_name}')
    
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss per Epoch for All Sheets')
#     plt.legend()
    
#     # Save the combined plot in both PNG and PDF formats
#     file_name = f"loss/loss_plot_all_{gen_name_unik}"
#     plt.savefig(f"{file_name}.png")
#     plt.savefig(f"{file_name}.pdf")
#     print(f"Loss plot all saved as PNG and PDF with unique name: {gen_name_unik}")

#     plt.close()  # Close the plot after saving
    
# Define plot_loss_all to plot loss for all sheets in one file
def plot_loss_all(loss_per_epoch_dict, sheets, gen_name_unik):
    plt.figure(figsize=(10, 6))
    
    #     for sheet_name, loss_per_epoch in zip(sheets, loss_per_epoch_dict):
    #         epochs = list(range(1, len(loss_per_epoch) + 1))
    #         # Adding legend with sheet name and final loss value formatted to 6 decimal places
    #         # plt.plot(epochs, loss_per_epoch, label=f"{sheet_name}+{loss_per_epoch[-1]:.6f}")
    #         # plt.plot(epochs, loss_per_epoch, label=f"{sheet_name} - {loss_per_epoch[-1]:.6f}")

    #         print(loss_per_epoch[-1])

    #         final_loss = float(loss_per_epoch[-1])  # Ensure this is a float
    #         # Adding legend with sheet name and final loss value formatted to 6 decimal places
    #         plt.plot(epochs, loss_per_epoch, label=f"{sheet_name} - {final_loss:.6f}")
        
    for sheet_name, loss_per_epoch in loss_per_epoch_dict.items():
        epochs = list(range(1, len(loss_per_epoch) + 1))
        final_loss = float(loss_per_epoch[-1])  # Ensure this is a float
        # Adding legend with sheet name and final loss value formatted to 6 decimal places
        plt.plot(epochs, loss_per_epoch, label=f"{sheet_name} - {final_loss:.16f}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch for All Sheets')
    plt.legend()
    
    # Save the combined plot in both PNG and PDF formats
    plt.savefig(f'loss/loss_plot_all_{gen_name_unik}.png')
    plt.savefig(f'loss/loss_plot_all_{gen_name_unik}.pdf')
    print(f"Loss plot all saved as PNG and PDF with identifier {gen_name_unik}")

    plt.close()  # Close the plot after saving


# Function to perform ELM regression for each sheet with dynamic hidden layers
def perform_elm_regression(file_path, sheets, hidden_layers, epochs=100):
    excel_file = pd.ExcelFile(file_path)
    
    gen_name_unik = datetime.today().astimezone(pytz.timezone('Asia/Jakarta')).strftime('%d-%m-%Y-%H-%M-%S')

    # untuk plot loss all
    # Collect loss data for all sheets
    loss_per_epoch_dict = {}
    
    for sheet_name in sheets:
        print(f"Processing sheet: {sheet_name}")
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        X = df.iloc[:, :n_input].values
        y = df.iloc[:, n_input:].values

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        elm_model = ELMRegression(n_input, hidden_layers, n_output)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(elm_model.parameters(), lr=0.01)
        loss_per_epoch = []

        elm_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = elm_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            loss_per_epoch.append(loss.item())

        
        save_model_json(elm_model, f"model_reg/model_{sheet_name}_{loss_per_epoch[-1]:.6f}_{len(hidden_layers)}_hidden_layers_{'_'.join(map(str, hidden_layers))}_{gen_name_unik}.json")
        save_loss_json(loss_per_epoch, f"loss/loss_{sheet_name}_{loss_per_epoch[-1]:.6f}_{len(hidden_layers)}_hidden_layers_{'_'.join(map(str, hidden_layers))}_{gen_name_unik}.json")
        
        plot_loss(loss_per_epoch, sheet_name, f"{loss_per_epoch[-1]:.6f}_{gen_name_unik}")
        print(f"Final Loss for {sheet_name}: {loss_per_epoch[-1]}")
        
        # existing code to train the model and get `loss_per_epoch`
        loss_per_epoch_dict[sheet_name] = loss_per_epoch  # Add to dictionary
        
    # untuk plot loss all
    # Collect loss data for all sheets
    #     loss_per_epoch_dict = {}

    #     for sheet_name in sheets:
    #         # existing code to train the model and get `loss_per_epoch`
    #         loss_per_epoch_dict[sheet_name] = loss_per_epoch  # Add to dictionary
        
    # After processing all sheets, call plot_loss_all
    plot_loss_all(loss_per_epoch_dict, sheets, gen_name_unik)
        
    
        
# Fungsi untuk meload model dari file JSON dengan support hidden layers yang dinamis
# def load_model_json(file_path, model, hidden_layers):
#     with open(file_path, 'r') as json_file:
#         model_params = json.load(json_file)

#     # Set parameter model dari file JSON untuk setiap hidden layer
#     for i, layer_size in enumerate(hidden_layers):
#         weight_key = f"hidden{i+1}_weights"
#         bias_key = f"hidden{i+1}_bias"
#         getattr(model, f"hidden{i+1}").weight.data = torch.FloatTensor(model_params[weight_key])
#         getattr(model, f"hidden{i+1}").bias.data = torch.FloatTensor(model_params[bias_key])
    
#     # Set parameter untuk output layer
#     model.output.weight.data = torch.FloatTensor(model_params['output_weights'])
#     model.output.bias.data = torch.FloatTensor(model_params['output_bias'])
    
#     print(f"Model loaded from {file_path}")
    
# Fungsi untuk meload model dari file JSON secara dinamis tanpa perlu hidden_layers sebagai parameter
def load_model_json(file_path, model):
    with open(file_path, 'r') as json_file:
        model_params = json.load(json_file)
    
    # Set parameter untuk setiap hidden layer
    i = 1
    while hasattr(model, f"hidden{i}"):
        weight_key = f"hidden{i}_weights"
        bias_key = f"hidden{i}_bias"
        getattr(model, f"hidden{i}").weight.data = torch.FloatTensor(model_params[weight_key])
        getattr(model, f"hidden{i}").bias.data = torch.FloatTensor(model_params[bias_key])
        i += 1
    
    # Set parameter untuk output layer
    model.output.weight.data = torch.FloatTensor(model_params['output_weights'])
    model.output.bias.data = torch.FloatTensor(model_params['output_bias'])
    
    print(f"Model loaded from {file_path}")
        
# Fungsi untuk menggabungkan semua sheet menjadi satu
def create_combined_sheet(file_path, sheets, combined_sheet_name="CombinedSheet"):
    excel_file = pd.ExcelFile(file_path)
    combined_df = pd.DataFrame()

    for sheet_name in sheets:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        combined_df.to_excel(writer, sheet_name=combined_sheet_name, index=False)
    
    print(f"Combined sheet '{combined_sheet_name}' created/overwritten in {file_path}")
    
# Fungsi untuk menguji model
def test_model(model, X_tensor, y_true):
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor)
        mse_loss = nn.MSELoss()(predictions, torch.FloatTensor(y_true))
        print(f'Testing Loss: {mse_loss.item()}')

# Usage example
file_path = 'dataset/dataset_v4.xlsx'
sheets = ['Sheet1-KM-SAR-Tiny-Reg', 'Sheet2-M-SAK-T-Tiny-Reg']

# name_CombinedSheet = 'x'.join(sheets)
name_CombinedSheet = 'KMT-Tiny-Reg'

# Buat sheet gabungan yang menggabungkan semua data dari sheet yang ada
create_combined_sheet(file_path, sheets)

# Lakukan regresi pada dataset di file Excel
# perform_elm_regression(file_path, sheets + ['CombinedSheet'], epochs=400)
# perform_elm_regression(file_path, sheets + ['Comb-'+name_CombinedSheet], epochs=400)

perform_elm_regression(file_path, sheets + ['Comb-'+name_CombinedSheet], hidden_layers, epochs=1000)
