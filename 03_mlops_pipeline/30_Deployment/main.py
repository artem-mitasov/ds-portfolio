import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def filter_data(df):
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)


def remove_year_outliers(df):
    df=df.copy()

    q25 = df['year'].quantile(0.25)
    q75 = df['year'].quantile(0.75)
    iqr = q75 - q25

    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    df['year'] = df['year'].clip(
        lower=round(lower_bound),
        upper=round(upper_bound)
    )
    return df

def add_short_model(df):
    df = df.copy()
    df['short_model'] = df['model'].apply(
        lambda x: x.lower().split(' ')[0] if pd.notna(x) else x
    )
    return df

def add_age_category(df):
    df = df.copy()
    df['age_category'] = df['year'].apply(
        lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average')
    )
    return df


def main():
    df = pd.read_csv('data/homework.csv')

    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('year_outliers', FunctionTransformer(remove_year_outliers)),
        ('short_model', FunctionTransformer(add_short_model)),
        ('age_category', FunctionTransformer(add_age_category)),
        ('columns', ColumnTransformer(transformers=[
            ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
            ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
        ]))
    ])

    models = (
        LogisticRegression(solver='lbfgs', max_iter=200),
        RandomForestClassifier(n_jobs=-1),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64), max_iter=700)
    )

    best_score = .0
    best_pipe = None

    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'\nbest model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'homework_30.pkl')


if __name__ == '__main__':(
    main())


