import pandas as pd
import numpy as np

def data_preprocessing(data_file):
    data = pd.read_csv(data_file)

    # 資料做中位數填補
    for col in ["bathrooms", "bedrooms", "livingRooms", "floorAreaSqM"]:
        data[col].fillna(data[col].median(), inplace=True) 

    # 填補類別補上缺失
    for col in ["tenure", "propertyType", "currentEnergyRating"]:
        data[col].fillna("Unknown", inplace=True)

    # 刪除不必要的欄位
    data = data.drop(columns=["ID", "fullAddress", "postcode"])

    data["price"] = np.log1p(data["price"])

    return data


if __name__ == "__main__":

    data_file = "train.csv"
    print(data_preprocessing(data_file))