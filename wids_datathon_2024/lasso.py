from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def apply_lasso(df_slice, target, cv=5):
    # Splitting data into features (X) and target variable (y)
    y = target
    X = df_slice

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the pipeline with LASSO regression
    lasso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso())
    ])

    # Define the parameter grid for alpha
    param_grid = {'lasso__alpha': [0.01, 0.1, 1, 10]}

    # Perform GridSearchCV
    grid_search = GridSearchCV(lasso_pipeline, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)

    # Display the best parameters and results
    print("Best Parameters for LASSO:", grid_search.best_params_)
    print("Best MSE for LASSO:", -grid_search.best_score_)

    # Get the best LASSO model from the grid search
    best_lasso_model = grid_search.best_estimator_

    # Extract important features
    lasso_coefficients = best_lasso_model.named_steps['lasso'].coef_
    important_features = X.columns[lasso_coefficients != 0]

    # Display the important features
    print("Important Features:", important_features)

    return list(important_features)