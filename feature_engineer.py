import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def feature_engineer(df:pd.DataFrame, target_col: str = "price", test_size= 0.2, random_state = 42):
    df = df.copy()

    num_features = ["bathrooms", "bedrooms", "livingRooms", "floorAreaSqM", "latitude", "longitude"]

    category_features = ["tenure", "propertyType", "currentEnergyRating", "country", "outcode"]

    X = df[num_features + category_features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, category_features),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)


    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "preprocessor": preprocessor,
        "num_features": num_features,
        "cat_features": category_features,
    }