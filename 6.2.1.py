import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# based on columns to drop, encoded columns, and columns that must not be na
def preprocess_data(df, columns_to_drop=None, columns_to_encode=None, na_subset=None):
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    if columns_to_encode:
        for col in columns_to_encode:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = df[col].astype('category').cat.codes
    if na_subset:
        df = df.dropna(subset=na_subset)
    return df

# try multiple max depths to find optimal through accuracy
def find_optimal_max_depth(X_train, y_train, X_test, y_test):
    best_accuracy = 0
    best_depth = None
    for depth in range(1, len(X_train.columns) + 1):
        tree = DecisionTreeClassifier(max_depth=depth)
        tree.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, tree.predict(X_test))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_depth = depth
    return best_depth


def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    y = y.loc[X.index]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Find the optimal max_depth
    optimal_max_depth = find_optimal_max_depth(X_train, y_train, X_test, y_test)
    
    # Train the model with the optimal max_depth
    tree = DecisionTreeClassifier(max_depth=optimal_max_depth)
    tree.fit(X_train, y_train)
    
    tree_rules = export_text(tree, feature_names=X.columns.tolist())
    num_leaf_nodes = tree.get_n_leaves()
    
    test_accuracy = accuracy_score(y_test, tree.predict(X_test))
    
    return test_accuracy, tree_rules, num_leaf_nodes, optimal_max_depth


# TITANIC DATASET
titanic = sns.load_dataset('titanic')
titanic_preprocessed = preprocess_data(
    titanic,
    columns_to_drop=['sibsp', 'parch', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive'],
    columns_to_encode=['sex', 'embarked'],
    na_subset=['embarked', 'age']
)
features = titanic_preprocessed.drop('survived', axis=1)
target = titanic_preprocessed['survived']

# Train the model and evaluate, getting the optimal max_depth as well
test_accuracy, tree_rules, num_leaf_nodes, optimal_max_depth = train_and_evaluate(features, target)

print(f"Test Accuracy: {test_accuracy}")
print(f"Optimal Max Depth: {optimal_max_depth}")
print("If-Then Rules:\n", tree_rules)
print(f"Number of If-Then Clauses: {num_leaf_nodes}")


