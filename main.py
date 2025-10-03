from data_cleaning_preprocess import data_preprocessing
from feature_engineer import feature_engineer
from build_model import train_model, evaluate_model
from predict import predict_and_save
if __name__ == "__main__":
    # 資料前處理
    data_file = "train.csv"
    data = data_preprocessing(data_file)
    data = feature_engineer(data)

    # 計算資料還有沒有缺額或空值
    #print(data.isnull().sum())

    X_train = data["X_train"].toarray()   # NN 要 numpy array
    X_test = data["X_test"].toarray()
    y_train = data["y_train"]
    y_test = data["y_test"]

    model, history = train_model(X_train, y_train,
                                 X_test, y_test,
                                 input_dim=X_train.shape[1])

    evaluate_model(model, X_test, y_test)
    preprocessor = data["preprocessor"]
    predict_and_save(
    model=model,
    preprocessor=preprocessor,   # 你訓練時用的同一個 preprocessor
    test_csv="test.csv",
    out_test_with_price="test_with_price.csv",
    out_submission="submission.csv",
    use_log_target=False         # 你現在沒有用 log 目標，保持 False
)

    