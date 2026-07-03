
import dill
import pandas as pd
import datetime

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
    import pandas as pd
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
    import pandas as pd
    df = df.copy()

    def calculate_boundaries(series):
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        return (round(q25 - 1.5 * iqr), round(q75 + 1.5 * iqr))

    lower, upper = calculate_boundaries(df['year'])

    df['year'] = df['year'].clip(lower=lower, upper=upper)
    return df


def add_short_model(df):
    import pandas as pd
    df = df.copy()
    df['short_model'] = df['model'].apply(
        lambda x: x.lower().split(' ')[0] if pd.notna(x) else x
    )
    return df

def add_age_category(df):
    import pandas as pd
    df = df.copy()
    df['age_category'] = df['year'].apply(
        lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average')
    )
    return df


def main():
    import pandas as pd
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
        print(
            f'model: {type(model).__name__}, '
            f'acc_mean: {score.mean():.4f}, '
            f'acc_std: {score.std():.4f}'
        )

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)

    metadata = {
        'name': 'Car price prediction model',
        'author': 'Artem Mitasov',
        'version': 1,
        'date': datetime.datetime.now().isoformat(),
        'type': type(best_pipe.named_steps['classifier']).__name__,
        'accuracy': round(best_score, 4)
    }

    print(f'\nbest model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    with open('cars_pipe.pkl', 'wb') as f:
        dill.dump(
            {
                'model': best_pipe,
                'metadata': metadata
            },
            f
        )


if __name__ == '__main__':
    main()

