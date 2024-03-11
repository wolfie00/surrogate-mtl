from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd


def get_adult(for_lime, one_hot):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race',
                    'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income']

    train_dataset = pd.read_csv('data_files/adult/adult.data', names=column_names, sep=',',
                                encoding='latin-1')
    train_df = train_dataset.dropna()

    test_dataset = pd.read_csv('data_files/adult/adult.test',
                               names=column_names, sep=',', encoding='latin-1',
                               #  dtype={'workclass': str}
                               )
    test_df = test_dataset.dropna()

    train_df = train_df.replace('(^\s+|\s+$)', '', regex=True)
    train_df = train_df.replace('?', np.nan)
    test_df = test_df.replace('(^\s+|\s+$)', '', regex=True)
    test_df = test_df.replace('?', np.nan)

    for col in ['workclass', 'occupation', 'native-country']:
        train_df[col].fillna(train_df[col].mode()[0], inplace=True)
        test_df[col].fillna(test_df[col].mode()[0], inplace=True)

    x_train = train_df.copy()
    x_test = test_df.copy()

    y_train = x_train.pop('income')
    y_test = x_test.pop('income')

    categorical_names = dict()
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
    # categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
    continuous_features = list(set(list(range(x_train.shape[1]))) - set(categorical_features))

    print('Loaded Adult dataset.')
    print('Categorical features:', categorical_features)
    print('Shape of training data (pre One-Hot Encoding):', x_train.shape)
    print('Shape of testing data (pre One-Hot Encoding):', x_test.shape)

    categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]

    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(x_train.iloc[:, feature])
        x_train.iloc[:, feature] = le.transform(x_train.iloc[:, feature])
        x_test.iloc[:, feature] = le.transform(x_test.iloc[:, feature])
        #     X_train[feature] = le.fit_transform(X_train[feature])
        #     X_test[feature] = le.transform(X_test[feature])
        categorical_names[feature] = le.classes_

    # for LIME
    x_train_exp = np.copy(x_train.values.astype(float))
    x_test_exp = np.copy(x_test.values.astype(float))

    ct = ColumnTransformer([
        ('scale', StandardScaler(), continuous_features),
        ('one_hot', OneHotEncoder(), categorical_features),
    ])

    x_train = ct.fit_transform(x_train.values.astype(float)).toarray()
    x_test = ct.transform(x_test.values.astype(float)).toarray()

    print('Shape of testing data (post One-Hot Encoding):', x_train.shape)
    print('Shape of testing data (post One-Hot Encoding):', x_test.shape)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
    #                                                     random_state = 42)

    y_train = y_train.map({'<=50K': 0, '>50K': 1})
    y_test = y_test.map({'<=50K.': 0, '>50K.': 1})

    if not for_lime:
        if one_hot:
            return x_train, y_train, x_test, y_test, None, None, None
        else:
            sc = StandardScaler()
            return sc.fit_transform(x_train_exp), y_train, sc.transform(x_test_exp), y_test, None, None, None
    else:
        return x_train_exp, y_train, x_test_exp, y_test, list(train_df.columns.values)[:-1], categorical_names, \
               ct if one_hot else None


def get_housing(test_size):
    from sklearn.datasets import fetch_california_housing

    california = fetch_california_housing()
    x, y = california.data, california.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=42)

    x_sc = StandardScaler()
    y_sc = StandardScaler()
    x_train = x_sc.fit_transform(np.array(x_train))
    x_test = x_sc.transform(np.array(x_test))
    y_train = y_sc.fit_transform(np.array(y_train).reshape(-1, 1)).flatten()
    y_test = y_sc.transform(np.array(y_test).reshape(-1, 1)).flatten()

    print('Loaded California Housing dataset. No categorical features.')
    print('Shape of training data:', x_train.shape)
    print('Shape of testing data:', x_test.shape)
    return x_train, y_train, x_test, y_test, california.feature_names, None, None


def get_autompg(test_size, for_lime):
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    autompg = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
    autompg = autompg.dropna()

    categorical_names = dict()
    categorical_names[6] = ['None', 'USA', 'Europe', 'Japan']

    encoded_dataset = autompg.copy()
    y = encoded_dataset.pop('MPG')

    x_train, x_test, y_train, y_test = train_test_split(encoded_dataset.values.astype(float),
                                                        y.values.astype(float), test_size=test_size,
                                                        random_state=42)

    x_train_exp = np.copy(x_train)
    x_test_exp = np.copy(x_test)

    print('Loaded AutoMPG dataset.')
    print('Categorical features:', ['Origin'])
    print('Shape of training data (pre One-Hot Encoding):', x_train.shape)
    print('Shape of testing data (pre One-Hot Encoding):', x_test.shape)

    ct = ColumnTransformer([
        ('scale', StandardScaler(), [0, 1, 2, 3, 4, 5]),
        ('one_hot', OneHotEncoder(), [6]),
    ])

    x_train = ct.fit_transform(x_train)
    x_test = ct.transform(x_test)

    print('Shape of testing data (post One-Hot Encoding):', x_train.shape)
    print('Shape of testing data (post One-Hot Encoding):', x_test.shape)

    y_sc = StandardScaler()
    y_train = y_sc.fit_transform(np.array(y_train).reshape(-1, 1)).flatten()
    y_test = y_sc.transform(np.array(y_test).reshape(-1, 1)).flatten()

    if not for_lime:
        return x_train, y_train, x_test, y_test, None, None, None
    else:
        return x_train_exp, y_train, x_test_exp, y_test, column_names[1:], categorical_names, ct


def get_titanic(test_size, for_lime):
    url = 'https://hbiostat.org/data/repo/titanic3.xls'
    # column_names = ['pclass', 'survived', 'name', 'sex', 'age', 'sibsp',
    #                 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest']

    df = pd.read_excel(url, )
    df.drop(['home.dest', 'name', 'body', 'boat', 'cabin', 'ticket'], inplace=True, axis=1)
    df['age'] = df['age'].fillna(df['age'].mean())
    df['embarked'] = df['embarked'].fillna(method='bfill')
    df['fare'] = df['fare'].fillna(df['fare'].mean())
    x = df.copy()
    y = x.pop('survived').values.astype(float)

    categorical_names = dict()
    categorical_features = [0, 1, 6]

    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(x.iloc[:, feature])
        x.iloc[:, feature] = le.transform(x.iloc[:, feature])
        categorical_names[feature] = le.classes_

    x_train, x_test, y_train, y_test = train_test_split(x.values.astype(float), y.flatten(), test_size=test_size,
                                                        random_state=42)

    print('Loaded Titanic dataset.')
    print('Categorical features:', ['pclass', 'sex', 'embarked'])
    print('Shape of training data (pre One-Hot Encoding):', x_train.shape)
    print('Shape of testing data (pre One-Hot Encoding):', x_test.shape)

    x_train_exp = np.copy(x_train)
    x_test_exp = np.copy(x_test)

    ct = ColumnTransformer([
        ('scale', StandardScaler(), [2, 3, 4, 5]),
        ('one_hot', OneHotEncoder(), [0, 1, 6]),
    ])

    x_train = ct.fit_transform(x_train)
    x_test = ct.transform(x_test)

    print('Shape of training data (post One-Hot Encoding):', x_train.shape)
    print('Shape of testing data (post One-Hot Encoding):', x_test.shape)

    if not for_lime:
        return x_train, y_train, x_test, y_test, None, None, None
    else:
        return x_train_exp, y_train, x_test_exp, y_test, list(x.columns.values), categorical_names, ct


def get_wine(test_size):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                    'quality']

    raw_dataset = pd.read_csv(url, sep=';')
    dataset = raw_dataset.dropna()

    train_size = 1. - test_size
    train_dataset = dataset.sample(frac=train_size, random_state=42)
    test_dataset = dataset.drop(train_dataset.index)
    x_train = train_dataset.copy()
    x_test = test_dataset.copy()

    y_train = x_train.pop('quality')
    y_test = x_test.pop('quality')

    x_sc = StandardScaler()
    y_sc = StandardScaler()
    x_train = x_sc.fit_transform(np.array(x_train.values))
    x_test = x_sc.transform(np.array(x_test.values))
    y_train = y_sc.fit_transform(np.array(y_train).reshape(-1, 1)).flatten()
    y_test = y_sc.transform(np.array(y_test).reshape(-1, 1)).flatten()

    print('Loaded Red Wine Quality dataset. No categorical features.')
    print('Shape of training data:', x_train.shape)
    print('Shape of testing data:', x_test.shape)
    return x_train, y_train, x_test, y_test, column_names[:-1], None, None


def get_data(name='housing', test_size=0.2, for_lime=False):
    if name == 'housing':
        return get_housing(test_size)
    elif name == 'adult':
        return get_adult(for_lime, one_hot=False)
    elif name == 'autompg':
        return get_autompg(test_size, for_lime)
    elif name == 'titanic':
        return get_titanic(test_size, for_lime)
    else:
        return get_wine(test_size)
