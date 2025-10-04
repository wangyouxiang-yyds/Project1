import pandas as pd
import numpy as np

def data_preprocessing(data_file):
    data = pd.read_csv(data_file)

    # 刪除不必要的欄位
    data = data.drop(columns=["ID", "fullAddress", "postcode", "country", "sale_month"])


    # 資料做中位數填補
    for col in ["bathrooms", "bedrooms", "livingRooms", "floorAreaSqM"]:
        data[col] = data[col].fillna(data[col].median())

    # 填補類別補上缺失
    for col in ["tenure", "propertyType", "currentEnergyRating"]:
        data[col] = data[col].fillna("Unknown")

    data["sale_year"] = 2025 - data["sale_year"]

    # price 做 log 結束後記得要補回來
    data["price"] = np.log1p(data["price"])
    print(f"data Nan in total :{data.isnull().sum()}")
    return data


if __name__ == "__main__":

    data_file = "train.csv"
    data_preprocessing(data_file)
    