import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
if __name__ == "__main__":
    data = pd.read_csv('dt.csv',header=None)
    print(data)
    imputer = SimpleImputer(strategy='most_frequent')  # Chọn chiến lược impute dữ liệu, ví dụ: thay thế bằng giá trị trung bình
    data = imputer.fit_transform(data)
    print(data)

