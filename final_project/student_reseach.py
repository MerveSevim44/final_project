
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
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

################################################
# 1. Exploratory Data Analysis
################################################
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    #print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

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

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def plot_categorical_distributions(df, cat_cols, palette="Set2"):
    """
    Verilen veri çerçevesi (df) içindeki kategorik değişkenlerin dağılımını çizer.

    Parametreler:
    df : pandas.DataFrame
        Veri çerçevesi
    cat_cols : list
        Kategorik değişken isimlerini içeren liste
    palette : str
        Seaborn renk paleti (varsayılan: "Set2")
    """
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df, palette=palette)
        plt.title(f"{col}  Kategorik Değişkeninin Dağılımı")
        plt.xlabel(col)
        plt.ylabel("Adet")
        plt.tight_layout()
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


def plot_numerical_histograms(df, num_cols, bins=15, color="mediumseagreen"):
    """
    Sayısal değişkenlerin histogramlarını çizer.

    Parametreler:
    df : pandas.DataFrame
        Veri çerçevesi
    num_cols : list
        Sayısal değişken isimlerini içeren liste
    bins : int
        Histogram için kullanılacak kutu (bin) sayısı (varsayılan: 15)
    color : str
        Histogram rengi (varsayılan: "mediumseagreen")
    """
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=bins, kde=True, color=color)
        plt.title(f"{col} Sayısal Değişkeninin Histogramı")
        plt.xlabel(col)
        plt.ylabel("Frekans")
        plt.tight_layout()
        plt.show()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def plot_target_mean_by_category(df, cat_cols, target, palette="muted"):
    """
    Kategorik değişkenlere göre hedef değişkenin ortalamasını gösteren barplot çizer.

    Parametreler:
    df : pandas.DataFrame
        Veri çerçevesi
    cat_cols : list
        Kategorik değişken isimlerini içeren liste
    target : str
        Hedef değişkenin adı
    palette : str
        Seaborn renk paleti (varsayılan: "muted")
    """
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.barplot(x=col, y=target, data=df, palette=palette, ci=None)
        plt.title(f"{target} Ortalaması ~ {col}")
        plt.xlabel(col)
        plt.ylabel(f"Ortalama {target}")
        plt.tight_layout()
        plt.show()


def plot_correlation_matrix(df, num_cols, cmap="RdBu", figsize=(12, 12), annot=True):
    """
    Sayısal değişkenler için korelasyon matrisini çizer.

    Parametreler:
    df : pandas.DataFrame
        Veri çerçevesi
    num_cols : list
        Sayısal değişken isimlerini içeren liste
    cmap : str
        Renk haritası (varsayılan: "RdBu")
    figsize : tuple
        Grafik boyutu (varsayılan: (12, 12))
    annot : bool
        Hücrelerdeki korelasyon değerlerinin gösterimi (varsayılan: True)
    """
    corr = df[num_cols].corr()

    sns.set(rc={'figure.figsize': figsize})
    sns.heatmap(corr, cmap=cmap, annot=annot, fmt=".2f", linewidths=0.5, square=True)
    plt.title("Korelasyon Matrisi", fontsize=16)
    plt.tight_layout()
    plt.show()

df = pd.read_csv("student-por.csv")
check_df(df)
cat_cols, cat_but_car, num_cols = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col)

plot_categorical_distributions(df,cat_cols)

for col in cat_cols:
    target_summary_with_cat(df,"G3",col)

plot_numerical_histograms(df,num_cols)

target_summary_with_cat(df,"G3",cat_cols)

plot_target_mean_by_category(df,cat_cols,"G3")

plot_correlation_matrix(df,num_cols)
################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def plot_feature_distributions(df, features, color="skyblue"):
    """
    Verilen özelliklerin histogramlarını çizer.

    Parametreler:
    df : pandas.DataFrame
        Veri çerçevesi
    features : list
        Görselleştirilecek özellik (değişken) isimlerinin listesi
    color : str
        Histogram rengi (varsayılan: "skyblue")
    """
    for col in features:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, kde=True, color=color)
        plt.title(f"{col} Dağılımı")
        plt.xlabel(col)
        plt.ylabel("Frekans")
        plt.tight_layout()
        plt.show()

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


for col in num_cols:
    if col != "G3":
      print(col, check_outlier(df, col))

for col in num_cols:
    if col != "G3":
        replace_with_thresholds(df,col)


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

    multi_class_cols = [col for col in df.columns if dataframe[col].dtype == "O" and dataframe[col].nunique() > 2]

    dataframe = one_hot_encoder(dataframe, multi_class_cols)

    y = dataframe['G3']  # Hedef değişken: Öğrencinin final sınavı puanı
    X = dataframe.drop(['G3'], axis=1)

    return X,y


df = pd.read_csv("student-por.csv")

check_df(df)

X, y = student_data_prep(df)

check_df(X)

######################################################
# 3. Base Models
######################################################
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


def plot_model_performance(model_names, rmse_scores):
    """
    Modellerin performans skorlarına göre barplot oluşturur.
    Performans skoru, 1 / RMSE ile hesaplanır.

    Parametreler:
    model_names : list
        Model isimlerinin listesi
    rmse_scores : list
        İlgili model için hesaplanan RMSE değerleri

    Dönüş:
    None
        Performansa göre sıralanmış barplot gösterir.
    """
    # Performans skorlarını hesapla
    performance_scores = [1 / score for score in rmse_scores]

    # DataFrame oluştur
    df_plot = pd.DataFrame({
        'Model': model_names,
        'RMSE': rmse_scores,
        'Performance (1/RMSE)': performance_scores
    })

    # Performansa göre sırala
    df_plot_sorted = df_plot.sort_values(by='Performance (1/RMSE)', ascending=False)

    # Barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Performance (1/RMSE)', data=df_plot_sorted, palette='viridis')

    plt.ylabel("Performans Skoru (1 / RMSE)")
    plt.xlabel("Model")
    plt.title("Modellere Göre Performans Karşılaştırması\n(Daha Yüksek Skor = Daha İyi)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

model_names = ['LR', 'KNN', 'CART', 'RF', 'GBM', 'XGBoost', 'LightGBM']
rmse_scores = [0.7561, 1.5211, 0.7599, 0.6231, 0.5823, 0.6366, 0.7146]

plot_model_performance(model_names, rmse_scores)

######################################################
# 4. Automated Hyperparameter Optimization
######################################################


regression_models = [
    ("GBM", GradientBoostingRegressor(), {
     "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 0.9, 1.0]
    })
]
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

best_models = hyperparameter_optimization_regression(X, y, regression_models)

######################################################
# 5. Stacking & Ensemble Learning
######################################################

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

voting_reg = voting_regressor(best_models, X, y)

######################################################
# 6. Prediction for a New Observation (ELde ettiğim modeli saklıyorum )
######################################################


X.columns
random_user = X.sample(1, random_state=45)
voting_reg.predict(random_user)
joblib.dump(voting_reg, "voting_reg.pkl")

new_model = joblib.load("voting_reg.pkl")
new_model.predict(random_user)






