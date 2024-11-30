import pandas as pd
import joblib

# Path ke model dan scaler
model_path = r'MAE11.pkl'  # Path ke model yang disimpan
scaler_path = r'D:\PowerStation\scalerMAE11.pkl'  # Path ke scaler yang disimpan

# Path input dan output file
input_file_path = r'ans6.csv'  # Pastikan file memiliki ekstensi .csv
output_file_path = r'C:\Users\B20_PC2\Downloads\PredictedPower.csv'

def parse_datetime_location(data_int):
    # Convert int to string for parsing
    data_string = str(data_int).zfill(14)  # Ensure zero-padding for consistency
    # Extract month, day, hour, and location code
    month = int(data_string[4:6])
    day = int(data_string[6:8])
    hour = int(data_string[8:10])
    location_code = int(data_string[12:])
    return month, day, hour, location_code

# Load model dan scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load data dari file input
new_data = pd.read_csv(input_file_path, header=0)

# Simpan kolom DateTime sebelum preprocessing
datetime_col = new_data['序號']

# Apply fungsi parse_datetime_location untuk setiap baris di kolom 'Serial'
new_data[['Month', 'Day', 'Hour', 'LocationCode']] = new_data['序號'].apply(parse_datetime_location).apply(pd.Series)

# Drop kolom Serial sebelum scaling dan encoding
new_data = new_data.drop(columns=['序號'])

# Scale numeric features
numeric_features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
new_data[numeric_features] = scaler.transform(new_data[numeric_features])

# Apply One-Hot Encoding for 'LocationCode'
new_data = pd.get_dummies(new_data, columns=['LocationCode'], drop_first=True)

# Pastikan kolom sama dengan model yang dilatih
required_features = model.feature_names_in_
new_data = new_data.reindex(columns=required_features, fill_value=0)

# Prediksi dengan model
predictions = model.predict(new_data)

# Pastikan hasil prediksi positif dan dua desimal
predictions = abs(predictions).round(2)

# Membuat DataFrame hasil prediksi
results = pd.DataFrame({
    '序號': datetime_col,
    'Power(mW)': predictions
})

# Simpan hasil prediksi ke file CSV
results.to_csv(output_file_path, index=False)

# Output informasi hasil
print(f"Hasil prediksi disimpan di: {output_file_path}")
