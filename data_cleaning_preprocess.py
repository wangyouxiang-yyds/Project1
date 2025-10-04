import pandas as pd
import numpy as np

def data_preprocessing(data_file, test_file):
    data_train = pd.read_csv(data_file)
    data_test = pd.read_csv(test_file)

    # 刪除不必要的欄位
    drop_columns = ["ID", "fullAddress", "postcode", "country", "sale_month"]
    data_train = data_train.drop(columns=drop_columns)
    data_test = data_test.drop(columns=drop_columns)


    # 資料做中位數填補
    for col in ["bathrooms", "bedrooms", "livingRooms", "floorAreaSqM"]:
        data_train[col] = data_train[col].fillna(data_train[col].median())
        data_test[col] = data_test[col].fillna(data_test[col].median())

    # 填補類別補上缺失
    for col in ["tenure", "propertyType", "currentEnergyRating"]:
        data_train[col] = data_train[col].fillna("Unknown")
        data_test[col] = data_test[col].fillna("Unknown")

    data_train["sale_year"] = 2025 - data_train["sale_year"]
    data_test["sale_year"] = 2025 - data_test["sale_year"]

    # price 做 log 結束後記得要補回來
    data_train["price"] = np.log1p(data_train["price"])
    print(f"train data Nan in total :{data_train.isnull().sum()}")
    print(f"test data Nan in total :{data_test.isnull().sum()}")
    return data_train, data_test


if __name__ == "__main__":

    data_file = "train.csv"
    test_file = "test.csv"
    data_preprocessing(data_file, test_file)
    