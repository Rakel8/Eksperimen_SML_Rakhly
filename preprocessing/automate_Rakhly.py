import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_preprocessing(input_path, output_path):
    print("Memulai proses otomatisasi data...")
    
    # 1. Memuat Dataset
    df = pd.read_csv(input_path)
    
    # 2. Membersihkan nama kolom
    df.columns = df.columns.str.strip()
    
    # 3. Menghapus kolom yang tidak relevan
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])
        
    # 4. Membersihkan spasi berlebih pada isi data teks
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
        
    # 5. Encoding
    le = LabelEncoder()
    categorical_cols = ['education', 'self_employed', 'loan_status']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            
    # 6. Memisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # 7. Scaling
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # 8. Menggabungkan kembali untuk disimpan
    df_processed = pd.concat([X, y], axis=1)
    
    # 9. Menyimpan hasil ke folder output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    
    print(f"Preprocessing Selesai! Data bersih disimpan di: {output_path}")

if __name__ == "__main__":
    # Menentukan lokasi file mentah dan lokasi tujuan file bersih
    RAW_DATA_PATH = "dataset_raw/loan_approval_dataset.csv"
    CLEAN_DATA_PATH = "loan_data_preprocessing/loan_data_clean.csv"
    
    run_preprocessing(RAW_DATA_PATH, CLEAN_DATA_PATH)