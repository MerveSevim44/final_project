################## STUDENT PERFORMANCE VERİ SETİ PROJESİ##################
##1.VERİ YÜKLEME VE İLK KEŞİF

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("student-por.csv")
df.head()
df.shape


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


check_df(df)

##2.VERİYE GENEL BAKIŞ VE KEŞİFSEL VERİ ANALİZİ(EDA)

def grab_col_names(dataframe, cat_th=10, car_th=20):
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

cat_cols, cat_but_car, num_cols = grab_col_names(df)



def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

for col in ["school", "sex", "famsize", "internet", "romantic"]:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df, palette="Set2")
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


for col in num_cols:
    num_summary(df, col, True)

num_cols = ["age", "absences", "G1", "G2", "G3"]

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=15, kde=True, color="mediumseagreen")
    plt.title(f"{col} Sayısal Değişkeninin Histogramı")
    plt.xlabel(col)
    plt.ylabel("Frekans")
    plt.tight_layout()
    plt.show()


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df,"G3",col)

cat_cols = ["school", "sex", "famsize", "romantic", "internet"]
target = "G3"

for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.barplot(x=col, y=target, data=df, palette="muted", ci=None)
    plt.title(f"{target} Ortalaması ~ {col}")
    plt.xlabel(col)
    plt.ylabel(f"Ortalama {target}")
    plt.tight_layout()
    plt.show()

#KORELASYON ANALİZİ
corr = df[num_cols].corr()
corr

# Korelasyonların gösterilmesi
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

##3.AYKIRI DEĞER ANALİZİ
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


for col in num_cols:
    if col != "G3":
      print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "G3":
        replace_with_thresholds(df,col)


##4.EKSİK VERİ ANALİZİ
df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)

##5.YENİ ÖZELLİKLER OLUŞTURMA(FEATURE ENGİNEERİNG)

df['NEW_internet_romantic_interaction'] = (df['internet'].map({"yes": 1, "no": 0}) * df['romantic'].map({"yes": 1, "no": 0}))+1
df['NEW_study_fail_interaction'] = df['studytime'] * df['failures']
df['NEW_higher_health_interaction'] = df['higher'].map({"yes": 1, "no": 0}) * df['health']
df['NEW_study_fail_time_socio'] = df['studytime'] * df['failures'] * df['famsize'].map({"LE3": 1, "GT3": 0})
df['NEW_school_famsize_interaction'] = df['school'] + "_" + df['famsize']
df['NEW_alc_health_interaction'] = df['Dalc'] * df['Walc'] * df['health']
df['NEW_famsup_G3_diff'] = df.groupby('famsup')['G3'].transform('mean')
df['NEW_school_avg'] = df.groupby('school')['G3'].transform('mean')


def categorize_grade(grade):
    if grade >= 15:
        return 'Başarılı'
    elif grade >= 10:
        return 'Ortalama'
    else:
        return 'Düşük'


df['NEW_G3_category'] = df['G3'].apply(categorize_grade)
df['NEW_avg_grade'] = (df['G1'] + df['G2']) / 2
df['NEW_total_parent_education'] = df['Medu'] + df['Fedu']
df['NEW_parent_education_effect_on_G3'] = (df['Medu'] + df['Fedu']) * df['G3']
df['NEW_parent_education_socio_interaction'] = (df['Medu'] + df['Fedu']) * (df['famsize'].map({"LE3": 1, "GT3": 0})+1)
df['NEW_parent_education_failures_interaction'] = (df['Medu'] + df['Fedu']) * (df['failures']+1)
df['NEW_social_support_success_interaction'] = (df['famsup'].map({'yes': 1, 'no': 0}) + df['schoolsup'].map({'yes': 1, 'no': 0})) * df['G3']
df['NEW_nursery_success_interaction'] = df['nursery'].map({'yes': 1, 'no': 0}) * df['G3']
df['NEW_reason_traveltime_interaction'] = df['reason'].map({'home': 1, 'reputation': 2, 'course': 3, 'other': 4}) * df['traveltime']
df['NEW_goout_traveltime_interaction'] = df['goout'] * df['traveltime']
df['NEW_famsize_schoolsup_interaction'] = (df['famsize'].map({'LE3': 1, 'GT3': 0}) * df['schoolsup'].map({'yes': 1, 'no': 0}))+1

new_features = ["NEW_avg_grade", "NEW_parent_education_effect_on_G3"]  # senin oluşturduğun sütunlar

for col in new_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, kde=True, color="skyblue")
    plt.title(f"{col} Dağılımı")
    plt.xlabel(col)
    plt.ylabel("Frekans")
    plt.tight_layout()
    plt.show()


df.columns = [col.upper() for col in df.columns]

df.head()

##6.KATEGORİK VERİLERİN KODLANMASI(ENCODİNG)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols =[col for col in df.columns if df[col].dtype not in [int,float]
              and df[col].nunique()==2]


for col in binary_cols:
    df=label_encoder(df,col)


df.head()

multi_class_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe,
                                columns=categorical_cols,
                                drop_first=drop_first,
                                dtype=int)
    return dataframe

df = one_hot_encoder(df, multi_class_cols)
df.head()

##7.MODELLEME VE DEĞERLENDİRME

y = df['G3']
X = df.drop(['G3'], axis=1)

# Model listesi
models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ('XGBoost', XGBRegressor(objective='reg:squarederror')),
          ('LightGBM', LGBMRegressor())]


# Model değerlendirmesi ve RMSE hesaplama
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")

#RMSE: 0.7561 (LR)
#RMSE: 1.5211 (KNN)
#RMSE: 0.7599 (CART)
#RMSE: 0.6231 (RF)
#RMSE: 0.5823 (GBM)
#RMSE: 0.6366 (XGBoost)
#RMSE: 0.7146 (LightGBM)

# Model isimleri ve RMSE değerleri
model_names = ['LR', 'KNN', 'CART', 'RF', 'GBM', 'XGBoost', 'LightGBM']
rmse_scores = [0.7561, 1.5211, 0.7599, 0.6231, 0.5823, 0.6366, 0.7146]

# RMSE yerine "performans skoru" hesaplıyoruz (ters orantı için)
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

# Gradient Boosting Modeli
gbm_model = GradientBoostingRegressor(random_state=99)

rmse = np.mean(np.sqrt(-cross_val_score(gbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print(f"Initial RMSE for GBM: {rmse}")


gbm_params = {
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 0.9, 1.0]
}


gbm_gs_best = GridSearchCV(
    gbm_model,
    gbm_params,
    cv=5,
    n_jobs=-1,
    verbose=True
).fit(X, y)

# En iyi parametreleri yazdırıyoruz
print("En iyi Gradient Boosting Model Parametreleri:", gbm_gs_best.best_params_)

# En iyi parametrelerle final modelini eğitiyoruz
final_gbm_model = gbm_model.set_params(**gbm_gs_best.best_params_).fit(X, y)

# Final modelin performansını tekrar cross-validation ile kontrol ediyoruz
final_rmse = np.mean(np.sqrt(-cross_val_score(final_gbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print(f"Final Gradient Boosting Model RMSE: {final_rmse}")


sample = X.iloc[[0]]
prediction = final_gbm_model.predict(sample)
print(f"Tahmin edilen G3: {round(prediction[0], 2)}")

### FEATURE IMPORTANCE
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num], palette="viridis")
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(gbm_model, X)

