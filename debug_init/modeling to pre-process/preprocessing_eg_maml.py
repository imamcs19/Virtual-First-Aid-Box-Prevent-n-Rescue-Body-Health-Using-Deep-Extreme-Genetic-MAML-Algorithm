n_input = 14   # Jumlah kolom yang akan diambil
n_output = 44
hidden_layers = [100, 50, 25, 12, 6, 3] 
input_file_path = "dataset/dataset_v4.xlsx"
sheet_name = 'CombinedSheet'

# Fungsi utama untuk memuat dan memproses data berdasarkan parameter yang diberikan
def get_data_test(id_test_data):
    # Memuat file Excel
    df = pd.read_excel(input_file_path, sheet_name=sheet_name)

    # Menghapus kolom "Unnamed: 0" jika ada
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Hapus semua kolom yang namanya mengandung 'Unnamed'

    # Ambil baris sesuai input_no_surah, dan 114 kolom terakhir
    # data_test_row = df.iloc[id_test_data, :n_input]  # Indeks dimulai dari 0, jadi kurangi 1
    # data_test = data_test_row.tolist()  # Mengonversi ke dalam bentuk list
    
    # Konversi menjadi list
    # data_test = data_test_row.values.tolist() if isinstance(data_test_row, pd.DataFrame) else data_test_row.tolist()
    
    # # Ambil data dari indeks tertentu atau banyak indeks (jika id_test_data berupa array)
    # if isinstance(id_test_data, (list, np.ndarray)):
    #     data_test_rows = df.iloc[id_test_data, :n_input]
    #     data_test = data_test_rows.values.tolist()  # Mengembalikan sebagai list of lists
    # else:
    #     data_test_row = df.iloc[id_test_data, :n_input]
    #     data_test = data_test_row.tolist()  # Mengembalikan sebagai list
    
    # Pastikan id_test_data adalah array 1 dimensi dari indeks
    if isinstance(id_test_data, (list, np.ndarray)):
        id_test_data = np.array(id_test_data).flatten()  # Konversi ke array 1D

    # Ambil data dari indeks tertentu atau banyak indeks (jika id_test_data berupa array)
    data_test_rows = df.iloc[id_test_data, :n_input]
    data_test = data_test_rows.values.tolist()  # Mengembalikan sebagai list of lists
    
    return data_test

def get_y_gt(id_test_data):
    # Memuat file Excel
    df = pd.read_excel(input_file_path, sheet_name=sheet_name)

    # Menghapus kolom "Unnamed: 0" jika ada
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Hapus semua kolom yang namanya mengandung 'Unnamed'

    # Ambil baris sesuai input_no_surah, dan 114 kolom terakhir
    y_gt_row = df.iloc[id_test_data, -n_output:]  # Indeks dimulai dari 0, jadi kurangi 1
    y_gt = y_gt_row.tolist()  # Mengonversi ke dalam bentuk list

    return y_gt

def test_single_data_return_loss(model, X_tensor, y_true):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_tensor))
        mse_loss = nn.MSELoss()(predictions, torch.FloatTensor(y_true))
        # print(f'Testing Loss: {mse_loss.item()}')
        
    return mse_loss.item()
        
def test_single_data_return_pred(model, single_data_test):
    model.eval()  # Set model ke mode evaluasi
    with torch.no_grad():  # Matikan gradient calculation
        input_tensor = torch.FloatTensor(single_data_test).unsqueeze(0)  # Tambahkan dimensi batch
        prediction = model(input_tensor)  # Lakukan prediksi
        # print(f"Data Uji: {single_data}")  # Tampilkan data uji
        # print(f"Hasil Regresi: {prediction.numpy().flatten()}")  # Tampilkan hasil regresi

    return prediction.numpy().flatten()

def get_topk_values_and_indices(predictions, topk):
    top_values, top_indices = torch.topk(torch.FloatTensor(predictions), topk)
    return top_values.numpy(), top_indices.numpy()

def get_top_k_columns(predictions, topk, n_input = 14, n_output = 44, input_file_path = "dataset/dataset_v4.xlsx", sheet_name = 'CombinedSheet'):
    # Baca file Excel
    df = pd.read_excel(input_file_path, sheet_name=sheet_name)

    # Ambil nama-nama kolom terakhir (n_output)
    output_column_names = df.iloc[0:0, -n_output:].columns

    # Dapatkan nilai dan indeks top-k
    top_values, top_indices = get_topk_values_and_indices(predictions, topk)

    # Ambil nama kolom berdasarkan indeks top-k dan konversi menjadi daftar string
    top_column_names = output_column_names[top_indices].tolist()

    # Return top values, indices, dan nama kolom
    return top_values, top_indices, top_column_names

id_test_data = 0
# test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]  # Contoh data uji
test_data = get_data_test(id_test_data)
print("Data Uji: ")  # Tampilkan data uji
print(test_data)
print('panjang fitur input = ',len(test_data))
print()

y_true_test_data = get_y_gt(id_test_data)

# Uji model dengan satu data uji tunggal
# test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]  # Contoh data uji
elm_model_n_hidden_layers = ELMRegression(n_input, hidden_layers, n_output)
# load_model_json(f'model_reg/model_Sheet1-KM-SAR-Tiny-Reg.json', elm_model)  # Ganti dengan nama sheet yang sesuai
# load_model_json(f'model_reg/model_Comb-KMT-Tiny-Reg_6_hidden_layers_100_50_25_12_6_3_31-10-2024-10-09-13.json', elm_model_n_hidden_layers, hidden_layers)  # Ganti dengan nama sheet yang sesuai
load_model_json(f'model_reg/model_Comb-KMT-Tiny-Reg_6_hidden_layers_100_50_25_12_6_3_31-10-2024-10-09-13.json', elm_model_n_hidden_layers)  # Ganti dengan nama sheet yang sesuai
hasil_pred = test_single_data_return_pred(elm_model_n_hidden_layers, test_data)
print(f"Hasil Regresi: {hasil_pred}") 
print(f"Panjang dim Hasil Regresi: {len(hasil_pred)}") 

print()
topk = 2
top_values, top_indices, top_column_names = get_top_k_columns(hasil_pred, topk)
print("Top Values:", top_values)
print("Top Indices:", top_indices)
print("Top Column Names:", top_column_names)

print()

# test_single_data_return_loss(elm_model, test_data, y)
# X_tensor = torch.FloatTensor(X)
# nilai_loss = test_single_data_return_loss(elm_model, torch.FloatTensor(test_data), np.array(y_true_test_data))
nilai_loss = test_single_data_return_loss(elm_model_n_hidden_layers, test_data, y_true_test_data)
print(f"Hasil nilai loss: {nilai_loss}") 