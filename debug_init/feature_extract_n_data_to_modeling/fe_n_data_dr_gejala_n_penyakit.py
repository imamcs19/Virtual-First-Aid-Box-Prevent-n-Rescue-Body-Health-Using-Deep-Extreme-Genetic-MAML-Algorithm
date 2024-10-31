import pandas as pd

# Provided data in separate lines
# data = [
#     "Biji-bijian Utuh dan Kacang-kacangan, Sayuan Berdaun Gelap, Telur, Daging Merah Segar",
#     "Jahe, Bayam, Kuning Telur, Tuna",
#     "Jeruk, Delima, Yoghurt, Madu, Telur, Ikan, Susu Kedelai, Daging (Sapi), Tiram, Hati Ayam, Kalkun, Wijen, Labu, Kentang, Brokoli, Air Kelapa",
#     "Sikat Kayu Siwak, Sup Hangat, Bubur, Jus Pisang, Jus Melon, Jus Blewah, Jus Tomat, Bubur Gandum dari Biji Food Grade (Oatmeal), Telur, Bubur Kacang Hijau, Smoothie Bowl (Mix Jus Bus Buah dan Chia Seed), Pasta",
#     "Air Putih, Wortel, Kubis, Kentang, Sup Ayam, Jus Delima, Pisang, Peppermint, Madu, Kunyit, Jahe, Teh chamomile, Bawang putih"
# ]

data = [
    "Migraine, Sakit kepala berdenyut di satu sisi",
    "Vertigo, Pusing berputar, kehilangan keseimbangan",
    "Sariawan (di mulut/lidah), Luka kecil pada mulut/lidah, nyeri di sekitar mulut/lidah/pipi/kepala",
    "Sakit Gigi, Nyeri di gigi atau gusi",
    "Radang Tenggorokan, Nyeri tenggorokan, batuk, suara serak"
]

# data = """
# 1. Biji-bijian Utuh dan Kacang-kacangan 2. Sayuan Berdaun Gelap 3. Telur 4. Daging Merah Segar
# 1. Jahe 2. Bayam 3. Kuning Telur 4. Tuna
# 1. Jeruk 2. Delima 3. Yoghurt 4. Madu 5. Telur 6. Ikan 7. Susu Kedelai 8. Daging (Sapi) 9. Tiram 10. Hati Ayam 11. Kalkun 12. Wijen 13. Labu 14. Kentang 15. Brokoli 16. Air Kelapa
# 1. Sikat Kayu Siwak 2. Sup Hangat 3. Bubur 4. Jus Pisang 5. Jus Melon 6. Jus Blewah 7. Jus Tomat 8. Bubur Gandum dari Biji Food Grade (Oatmeal) 9. Telur 10. Bubur Kacang Hijau 11. Smoothie Bowl (Mix Jus Bus Buah dan Chia Seed) 12. Pasta
# 1. Air Putih 2. Wortel 3. Kubis 4. Kentang 5. Sup Ayam 6. Jus Delima 7. Pisang 8. Peppermint 9. Madu 10. Kunyit 11. Jahe 12. Teh chamomile 13. Bawang putih
# """

# Fungsi untuk menghitung banyak kata
def count_words(data):
    word_count = 0
    for entry in data:
        # Memisahkan setiap string berdasarkan koma
        # print(entry)
        parts = entry.split(',')
        # print(parts)
        word_count += len(parts)
        # for part in parts:
        #     # Menghitung jumlah kata dalam setiap bagian
        #     word_count += len(part.strip().split())
        #     print(word_count)
    return word_count
print('Panjang data awal = ', count_words(data))

# Split each row into individual items and strip any extra whitespace
rows = [set(item.strip().lower() for item in row.split(',')) for row in data]

# print(rows)

print('baris data = ', len(rows))

# Get all unique items from all rows
unique_items = sorted(set(item for row in rows for item in row))

print('panjang data unik = ', len(unique_items))

# Create a DataFrame with rows of data, where each column is a unique item (one-hot encoded)
one_hot_data = []
for row in rows:
    one_hot_data.append([1 if item in row else 0 for item in unique_items])

# Create the table with pandas
df = pd.DataFrame(one_hot_data, columns=unique_items)

# Display the resulting table
display(df)

# # Save the resulting table to an Excel file
# output_file = "dataset/one_hot_encoded_data.xlsx"
# df.to_excel(output_file, index=False)

# print(f"The one-hot encoded data has been saved to {output_file}")

# Simpan tabel hasil ke file Excel dengan nama sheet tertentu
output_file = "dataset/one_hot_encoded_data.xlsx"
sheet_name = "OneHotEncodedDataInput"  # Nama sheet yang diinginkan

# # Simpan DataFrame ke Excel dengan nama sheet tertentu
# with pd.ExcelWriter(output_file) as writer:
#     df.to_excel(writer, index=False, sheet_name=sheet_name)

# print(f"Data one-hot encoded telah disimpan ke {output_file} pada sheet '{sheet_name}'")

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
