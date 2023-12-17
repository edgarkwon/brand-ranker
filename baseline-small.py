import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


# Load the data
store_df = pd.read_csv('./data/store.csv')
brand_df = pd.read_csv('./data/brand.csv')
area_df = pd.read_csv('./data/area-aggregated.csv')

# Merge the dataframes
merged_df = store_df.merge(brand_df, on='brand_name').merge(area_df, on='area_code')

# Continue with feature engineering, data splitting, model training, and evaluation...
features = ["당월_매출_금액","당월_매출_건수","월요일_매출_비율","토요일_매출_비율","시간대_11~14_매출_비율","시간대_17~21_매출_비율","연령대_10_매출_비율","연령대_20_매출_비율","연령대_30_매출_비율","연령대_40_매출_비율","남성_매출_비율"
,"월간 총 검색량","연별 검색량-2021년","연별 검색량-2022년","요일비율-월","연령별검색비율 - 10대 ","연령별검색비율 - 20대 ","연령별검색비율 - 30대 ","연령별검색비율 - 40대 ","성별검색비율 - 남성","이슈성"
]

# Impute missing values
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', 'constant', etc.
X = imputer.fit_transform(merged_df[features])
y = merged_df['score']

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
model = LinearRegression()

# Model Training
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')