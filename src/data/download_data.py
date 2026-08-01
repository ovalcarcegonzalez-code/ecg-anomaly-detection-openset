import wfdb
import os

os.makedirs('data/raw/mitdb', exist_ok=True)
wfdb.dl_database('mitdb', dl_dir='data/raw/mitdb')
print("Descarga completada")
