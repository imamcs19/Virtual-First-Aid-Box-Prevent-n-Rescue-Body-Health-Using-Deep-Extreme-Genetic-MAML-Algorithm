## n_hidden = 3 (n_hidden = 3 as synthetic reptile < n_hidden = 6 pada model as GT)

import json
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

   
# untuk support 
# for param in elm_model_reptile.parameters():
#     print(param.data)
#     print(param.grad.data)

class ModelForSyntheticReptile(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(ModelForSyntheticReptile, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.output = nn.Linear(n_hidden3, n_output)

        # Remove the following lines if you need gradients for hidden layers
        # for param in self.hidden1.parameters():
        #     param.requires_grad = False
        # for param in self.hidden2.parameters():
        #     param.requires_grad = False
        # for param in self.hidden3.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        x = self.output(x)
        return x

def save_model_elm_reptile_to_json(model, file_path):
    model_dict = {
        "architecture": {
            "n_input": model.hidden1.in_features,
            "n_hidden1": model.hidden1.out_features,
            "n_hidden2": model.hidden2.out_features,
            "n_hidden3": model.hidden3.out_features,
            "n_output": model.output.out_features,
        },
        "state_dict": {k: v.tolist() for k, v in model.state_dict().items()}
    }
    with open(file_path, 'w') as f:
        json.dump(model_dict, f)
    print(f"Model saved to {file_path}")

def load_model_elm_reptile_from_json(file_path):
    with open(file_path, 'r') as f:
        model_dict = json.load(f)

    arch = model_dict["architecture"]
    model = ELMRegressionForReptile(
        n_input=arch["n_input"],
        n_hidden1=arch["n_hidden1"],
        n_hidden2=arch["n_hidden2"],
        n_hidden3=arch["n_hidden3"],
        n_output=arch["n_output"]
    )

    state_dict = OrderedDict({k: torch.tensor(v) for k, v in model_dict["state_dict"].items()})
    model.load_state_dict(state_dict)
    print(f"Model loaded from {file_path}")
    
    return model

# Helper function to save loss to a JSON file
def save_loss_model_elm_reptile_to_json(loss_per_epoch, file_path):
    with open(file_path, 'w') as f:
        json.dump(loss_per_epoch, f)
    print(f"Loss saved to {file_path}")

# Helper function to plot loss and save as PNG and PDF
def plot_loss_model_elm_reptile(loss_per_epoch, sheet_name):
    plt.plot(loss_per_epoch)
    plt.title(f'Loss per Epoch for {sheet_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    
    png_path = f'loss/loss_plot_reptile_{sheet_name}.png'
    pdf_path = f'loss/loss_plot_reptile_{sheet_name}.pdf'
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Loss plot saved to {png_path} and {pdf_path}")

# Main function to perform ELM regression using Reptile
def perform_elm_regression_reptile(file_path, sheets, epochs=100):
    excel_file = pd.ExcelFile(file_path)

    for sheet_name in sheets:
        print(f"Processing sheet: {sheet_name}")
        
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        X = df.iloc[:, :n_input].values  
        y = df.iloc[:, n_input:].values  

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        elm_model_reptile = ModelForSyntheticReptile(n_input, n_hidden1, n_hidden2, n_hidden3, n_output)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(elm_model_reptile.parameters(), lr=0.01)

        # List to store loss per epoch
        loss_per_epoch = []

        # Training model with data
        elm_model_reptile.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = elm_model_reptile(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            loss_per_epoch.append(loss.item())

        # Save model and loss
        save_model_elm_reptile_to_json(elm_model_reptile, f'model_reg/model_reptile_{sheet_name}.json')
        save_loss_model_elm_reptile_to_json(loss_per_epoch, f'loss/loss_reptile_{sheet_name}.json')

        # Plot and save loss plot
        plot_loss_model_elm_reptile(loss_per_epoch, sheet_name)

        print(f"Final Loss for {sheet_name}: {loss_per_epoch[-1]}")

# Parameters
n_input = 14
n_hidden1 = 100
n_hidden2 = 50
n_hidden3 = 25
n_output = 44

# Example usage
# file_path = 'dataset/dataset_v4.xlsx'
# sheets = ['Sheet1', 'Sheet2']  # List of sheet names to process
# perform_elm_regression_reptile(file_path, sheets, epochs=100)

file_path = 'dataset/dataset_v4.xlsx'

# sheets = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4']
sheets = ['Sheet1-KM-SAR-Tiny-Reg', 'Sheet2-M-SAK-T-Tiny-Reg']

#generate kode unik

# name_CombinedSheet = 'x'.join(sheets)
name_CombinedSheet = 'KMT-Tiny-Reg'

# Buat sheet gabungan yang menggabungkan semua data dari sheet yang ada
create_combined_sheet(file_path, sheets)

# Lakukan regresi pada dataset di file Excel
# perform_elm_regression(file_path, sheets + ['CombinedSheet'], epochs=400)
perform_elm_regression_reptile(file_path, sheets + ['Comb-'+name_CombinedSheet], epochs=1000)
