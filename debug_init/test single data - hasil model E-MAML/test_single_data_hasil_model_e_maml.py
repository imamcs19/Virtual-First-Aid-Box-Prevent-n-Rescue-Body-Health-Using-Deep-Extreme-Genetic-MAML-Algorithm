# declare params
n_input = 14
n_hidden1 = 100
n_hidden2 = 50
n_hidden3 = 25
n_output = 44
input_file_path = "dataset/dataset_v4.xlsx"
sheet_name = 'CombinedSheet'
path_last_Model = 'model_reg_last/model_last_E-MAML_0.247_31-10-2024-20-45-23.json'
# path_last_Model = 'model_reg_last/model_last_E-MAML_Synthetic_E-Reptile_0.248_31-10-2024-20-44-27.json'


# model = ELMRegressionForReptile(n_input, n_hidden1, n_hidden2, n_hidden3, n_output)
model = ModelForSyntheticReptile(n_input, n_hidden1, n_hidden2, n_hidden3, n_output)

try:
    load_model_reptile_checkpoint(model, path_last_Model)
    print("Model berhasil dimuat dari:", path_last_Model)
except Exception as e:
    print("Terjadi kesalahan saat memuat model:", e)

# Sekarang Anda bisa menggunakan model untuk melakukan prediksi atau melanjutkan pelatihan
# Contoh prediksi
# test_data = to_torch([[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]])  # Ganti dengan data yang sesuai

id_test_data = 0
# test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]  # Contoh data uji
test_data = get_data_test(id_test_data)
print("Data Uji: ")  # Tampilkan data uji
print(test_data)
print('panjang fitur input = ',len(test_data))
print()

y_true_test_data = get_y_gt(id_test_data)

# model.eval()  # Set model ke mode evaluasi
# with torch.no_grad():
#     prediction = model(x_test)
#     print("Hasil prediksi:", prediction.numpy())
    

hasil_pred = test_single_data_return_pred(model, test_data)
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
nilai_loss = test_single_data_return_loss(model, test_data, y_true_test_data)
print(f"Hasil nilai loss: {nilai_loss}") 