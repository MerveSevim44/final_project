import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import GridSearchCV, cross_validate
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV


def grab_col_names(dataframe, cat_th=7, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols

def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe,
                                columns=categorical_cols,
                                drop_first=drop_first,
                                dtype=int)
    return dataframe  # BU SATIR EKLENMELİ

def student_data_prep(dataframe):

    #dataframe['NEW_internet_romantic_interaction'] = (dataframe['internet'].map({"yes": 1, "no": 0}) * dataframe['romantic'].map(
     #   {"yes": 1, "no": 0})) + 1
    dataframe['NEW_study_fail_interaction'] = dataframe['studytime'] * dataframe['failures']
    #dataframe['NEW_higher_health_interaction'] = dataframe['higher'].map({"yes": 1, "no": 0}) * dataframe['health']
    #dataframe['NEW_study_fail_time_socio'] = dataframe['studytime'] * dataframe['failures'] * dataframe['famsize'].map({"LE3": 1, "GT3": 0})
    #dataframe['NEW_school_famsize_interaction'] = dataframe['school'] + "_" + dataframe['famsize']
    dataframe['NEW_alc_health_interaction'] = dataframe['Dalc'] * dataframe['Walc'] * dataframe['health']
    dataframe['NEW_famsup_G3_diff'] = dataframe.groupby('famsup')['G3'].transform('mean')
    dataframe['NEW_school_avg'] = dataframe.groupby('school')['G3'].transform('mean')

    # Kategorik G3 tanımlaması
    def categorize_grade(grade):
        if grade >= 15:
            return 'Başarılı'
        elif grade >= 10:
            return 'Ortalama'
        else:
            return 'Düşük'

    dataframe['NEW_G3_category'] = dataframe['G3'].apply(categorize_grade)

    # Diğer yeni değişkenler
    dataframe['NEW_avg_grade'] = (dataframe['G1'] + dataframe['G2']) / 2
    dataframe['NEW_total_parent_education'] = dataframe['Medu'] + dataframe['Fedu']
    dataframe['NEW_parent_education_effect_on_G3'] = (dataframe['Medu'] + dataframe['Fedu']) * dataframe['G3']
    dataframe['NEW_parent_education_socio_interaction'] = (dataframe['Medu'] + dataframe['Fedu']) * (
                dataframe['famsize'].map({"LE3": 1, "GT3": 0}) + 1)
    dataframe['NEW_parent_education_failures_interaction'] = (dataframe['Medu'] + dataframe['Fedu']) * (dataframe['failures'] + 1)
    dataframe['NEW_social_support_success_interaction'] = (dataframe['famsup'].map({'yes': 1, 'no': 0}) + dataframe['schoolsup'].map(
        {'yes': 1, 'no': 0})) * dataframe['G3']
    dataframe['NEW_nursery_success_interaction'] = dataframe['nursery'].map({'yes': 1, 'no': 0}) * dataframe['G3']
    dataframe['NEW_reason_traveltime_interaction'] = dataframe['reason'].map({'home': 1, 'reputation': 2, 'course': 3, 'other': 4}) * \
                                              dataframe['traveltime']
    dataframe['NEW_goout_traveltime_interaction'] = dataframe['goout'] * dataframe['traveltime']
    dataframe['NEW_famsize_schoolsup_interaction'] = (dataframe['famsize'].map({'LE3': 1, 'GT3': 0}) * dataframe['schoolsup'].map(
        {'yes': 1, 'no': 0})) + 1

    # Eklenen yeni sütunları listele
    new_features = [col for col in dataframe.columns if col.startswith("NEW_")]

    """""
    for col in cat_cols:
        cat_summary(dataframe, col)

    for col in num_cols:
        num_summary(dataframe, col)
    """
    #plot_feature_distributions(dataframe, new_features)

    dataframe.head()

    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
                   and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    multi_class_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O" and dataframe[col].nunique() > 2]

    dataframe = one_hot_encoder(dataframe, multi_class_cols)

    y = dataframe['G3']  # Hedef değişken: Öğrencinin final sınavı puanı
    X = dataframe.drop(['G3'], axis=1)

    return X,y


def evaluate_models(models, X, y):
    """
    Verilen regresyon modellerini kullanarak çapraz doğrulama yapar ve her model için RMSE'yi hesaplar.

    Parametreler:
    models : list
        Kullanılacak regresyon modellerinin listesi (model ismi ve nesnesi tuple olarak)
    X : pandas.DataFrame veya numpy.ndarray
        Özellikler (input)
    y : pandas.Series veya numpy.ndarray
        Hedef değişken (output)

    Dönüş:
    None
        Her modelin RMSE değerini yazdırır.

    """
    models = [('LR', LinearRegression()),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('GBM', GradientBoostingRegressor()),
              ('XGBoost', XGBRegressor(objective='reg:squarederror')),
              ('LightGBM', LGBMRegressor())]

    for name, regressor in models:
        # RMSE hesaplama
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name})")

regression_models = [
    ("GBM", GradientBoostingRegressor(), {
     "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 0.9, 1.0]
    })]

def hyperparameter_optimization_regression(X, y, models, cv=5, scoring="neg_mean_squared_error"):
    print("Hyperparameter Optimization....")
    best_models = {}

    for name, regressor, params in models:
        print(f"########## {name} ##########")

        # İlk değerlendirme (optimizasyon öncesi)
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        rmse_before = np.mean(np.sqrt(-cv_results['test_score']))
        print(f"RMSE (Before): {round(rmse_before, 4)}")

        # GridSearchCV ile hiperparametre ayarı
        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)

        # En iyi parametrelerle final model
        final_model = regressor.set_params(**gs_best.best_params_)
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        rmse_after = np.mean(np.sqrt(-cv_results['test_score']))
        print(f"RMSE (After): {round(rmse_after, 4)}")
        print(f"{name} best params: {gs_best.best_params_}\n")

        best_models[name] = final_model

    return best_models

def voting_regressor(best_models, X, y, cv=5):
    print("Voting Regressor...")

    # VotingRegressor'a uygun modelleri belirle
    voting_reg = VotingRegressor(estimators=[
        ('GBM', best_models["GBM"])
    ]).fit(X, y)

    # RMSE hesaplamak için negatif MSE skoru üzerinden dönüşüm
    cv_results = cross_validate(voting_reg, X, y, cv=cv, scoring="neg_mean_squared_error")
    rmse = np.mean(np.sqrt(-cv_results['test_score']))

    print(f"RMSE (Voting Regressor): {round(rmse, 4)}")

    return voting_reg

def main():
    df = pd.read_csv("student-por.csv")
    X,y = student_data_prep(df)
    evaluate_models(regression_models,X,y)
    best_models = hyperparameter_optimization_regression(X,y,regression_models)
    voting_reg = voting_regressor(best_models, X, y)
    joblib.dump(voting_reg, "voting_reg1.pkl")
    return voting_reg


if __name__ == "__main__":
    print("İşlem başladı")
    main()




