import re
import pandas as pd

# Data awal
data = """
1. Biji-bijian Utuh dan Kacang-kacangan 2. Sayuan Berdaun Gelap 3. Telur 4. Daging Merah Segar
1. Jahe 2. Bayam 3. Kuning Telur 4. Tuna
1. Jeruk 2. Delima 3. Yoghurt 4. Madu 5. Telur 6. Ikan 7. Susu Kedelai 8. Daging (Sapi) 9. Tiram 10. Hati Ayam 11. Kalkun 12. Wijen 13. Labu 14. Kentang 15. Brokoli 16. Air Kelapa
1. Sikat Kayu Siwak 2. Sup Hangat 3. Bubur 4. Jus Pisang 5. Jus Melon 6. Jus Blewah 7. Jus Tomat 8. Bubur Gandum dari Biji Food Grade (Oatmeal) 9. Telur 10. Bubur Kacang Hijau 11. Smoothie Bowl (Mix Jus Bus Buah dan Chia Seed) 12. Pasta
1. Air Putih 2. Wortel 3. Kubis 4. Kentang 5. Sup Ayam 6. Jus Delima 7. Pisang 8. Peppermint 9. Madu 10. Kunyit 11. Jahe 12. Teh chamomile 13. Bawang putih
"""

# Step 1: Hapus "number + dot" di awal baris sebelum spasi
cleaned_data = re.sub(r'^\d+\.\s', '', data, flags=re.MULTILINE)

# Step 2: Split baris menjadi list untuk memanipulasi tiap baris
lines = cleaned_data.strip().split('\n')

# Step 3: Tambahkan ",\n" di akhir setiap baris kecuali baris terakhir
formatted_data = ",\n".join(lines) + '\n'

# Step: Replace "number + dot" with comma
cleaned_data2 = re.sub(r'\d+\.\s', ', ', cleaned_data)

# Replace newline characters with commas
cleaned_data3 = re.sub(r'\n', ', ', cleaned_data2)

# Step 4: Split the data by comma to count elements
split_data = [item.strip() for item in cleaned_data3.split(',') if item.strip()]

# Hitung jumlah elemen
count = len(split_data)
print('Panjang data awal = ', count)

# Step 5: Split cleaned_data into a list of lines
lines = cleaned_data2.strip().split('\n')

# Step 6: Store each line as a string in a list
lines_as_strings = [line.strip() for line in lines]

# Split each row into individual items and strip any extra whitespace
rows = [set(item.strip().lower() for item in row.split(',')) for row in lines_as_strings]

# Hitung jumlah baris data
print('Jumlah baris data = ', len(rows))

# Ambil semua item unik dari semua baris
unique_items = sorted(set(item for row in rows for item in row))
print('Panjang data unik = ', len(unique_items))

# Buat DataFrame dengan one-hot encoding
one_hot_data = []
for row in rows:
    one_hot_data.append([1 if item in row else 0 for item in unique_items])

# Buat tabel dengan pandas
df = pd.DataFrame(one_hot_data, columns=unique_items)

# Tampilkan tabel yang dihasilkan
display(df)

# Simpan tabel hasil ke file Excel dengan nama sheet tertentu
output_file = "dataset/one_hot_encoded_data.xlsx"
sheet_name = "OneHotEncodedDataTarget"  # Nama sheet yang diinginkan

# Cek apakah file sudah ada dan append data jika ada
try:
    with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    print(f"Data one-hot encoded telah ditambahkan ke {output_file} pada sheet '{sheet_name}'")
except FileNotFoundError:
    # Jika file tidak ada, simpan sebagai file baru
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    print(f"File {output_file} telah dibuat dan data ditambahkan pada sheet '{sheet_name}'")
