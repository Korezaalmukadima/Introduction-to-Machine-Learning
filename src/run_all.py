"""Jalankan semua studi kasus secara berurutan:  python src/run_all.py"""
import importlib

MODULES = [
    "01_house_price_regression",
    "02_loan_classification",
    "03_customer_segmentation",
    "04_anomaly_detection",
]

if __name__ == "__main__":
    for name in MODULES:
        importlib.import_module(name).main()
    print("\nSelesai. Semua gambar ada di folder outputs/")
