# === 放在 evaluate_model 定義「下面」 ===
import pandas as pd
import numpy as np

# 和訓練時一致的特徵欄位（不做衍生）
NUM_FEATS = ["bathrooms", "bedrooms", "livingRooms", "floorAreaSqM", "latitude", "longitude"]
CAT_FEATS = ["tenure", "propertyType", "currentEnergyRating", "country", "outcode"]

def predict_and_save(model, preprocessor,
                     test_csv="test.csv",
                     out_test_with_price="test_with_price.csv",
                     out_submission="submission.csv",
                     use_log_target=False):
    """
    讀 test.csv -> 預測 price -> 輸出:
      1) test_with_price.csv  (原 test 多最後一欄 price)
      2) submission.csv       (只含 ID, price)
    """
    # 1) 讀取 test
    test = pd.read_csv(test_csv)

    # 2) 準備特徵並做前處理（與訓練一致）
    X_test = test[NUM_FEATS + CAT_FEATS]
    X_test_pp = preprocessor.transform(X_test)
    # Keras 需要 dense，若是 sparse 就轉 dense
    if hasattr(X_test_pp, "toarray"):
        X_test_pp = X_test_pp.toarray()

    # 3) 預測（你現在不用 log 目標，就直接輸出）
    y_pred = model.predict(X_test_pp)
    # 兼容 sklearn 與 keras：可能會回 (n,1) 或 (n,)
    y_pred = np.asarray(y_pred).reshape(-1)
    if use_log_target:
        y_pred = np.expm1(y_pred)

    # 4) 寫回 test 最後一欄 price
    test_out = test.copy()
    test_out["price"] = y_pred
    test_out.to_csv(out_test_with_price, index=False)

    # 5) 只輸出 ID + price 作為提交檔
    submission = test_out[["ID", "price"]].copy()
    submission.to_csv(out_submission, index=False)

    print(f"Saved: {out_test_with_price} & {out_submission}")
    print(submission.head())
