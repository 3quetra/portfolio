from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def train_test_models(train, target):      
    # Splitting data into features (X) and target variable (y)
    y = target
    X = train

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    # Creating a pipeline with different classifiers and scalers
    pipelines = {
    'RandomForest': Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),

    'SVM': Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', SVC(random_state=42))
    ]),

    'KNN': Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', KNeighborsClassifier())
    ])
    }

    # Define the parameter grids for each classifier
    param_grid = {
    'RandomForest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': [0.1, 0.01, 0.001],
    },
    'KNN': {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance'],
    }
    }

    # Dictionary to store feature importances for each model
    feature_importances_dict = {}

    # Training and evaluating each model with GridSearchCV
    for name, pipeline in pipelines.items():
        grid_search = GridSearchCV(pipeline, param_grid=param_grid[name], cv=5, scoring='accuracy', n_jobs=-1)

        print(f"\nTraining {name} model with GridSearchCV...")
        grid_search.fit(X_train, y_train)

        # Display the best parameters and results
        print(f"Best Parameters for {name}:", grid_search.best_params_)
        print(f"Best Accuracy for {name}:", grid_search.best_score_)

        # Get the best model from the grid search
        best_model = grid_search.best_estimator_

        # Extract feature importances if the model is RandomForestClassifier
        if name == 'RandomForest':
            random_forest_model = best_model.named_steps['classifier']
            feature_importances_dict[name] = dict(zip(X_train.columns, random_forest_model.feature_importances_))
        else:
            feature_importances_dict[name] = None

        # Evaluate the best model on the test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"\nBest {name} Model Evaluation:\n{accuracy}\n{report}")

    return best_model, feature_importances_dict


def feature_importance(feature_importances_dict):
    # Display feature importances
    for name, feature_importances in feature_importances_dict.items():
        print(f"\nFeature Importances for {name} model:")
        if feature_importances:
            for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
                print(f"{feature}: {importance}")
        else:
            print(f"No feature importances available for {name} model.")
