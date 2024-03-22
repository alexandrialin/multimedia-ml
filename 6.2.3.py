import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

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
    # Ensure y has the same index as X for alignment (had an error in this dataset?)
    y = y.loc[X.index]
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    optimal_max_depth = find_optimal_max_depth(X_train, y_train, X_test, y_test)
    
    # Train the model with the optimal max_depth
    tree = DecisionTreeClassifier(max_depth=optimal_max_depth)
    tree.fit(X_train, y_train)
    
    tree_rules = export_text(tree, feature_names=X.columns.tolist())
    num_leaf_nodes = tree.get_n_leaves()    
    test_accuracy = accuracy_score(y_test, tree.predict(X_test))
    
    return test_accuracy, tree_rules, num_leaf_nodes, optimal_max_depth



def create_artificial_dataset(num_samples=1000, num_features=10):
    # Generate random feature values
    X = np.random.rand(num_samples, num_features)
    # Generate binary target values
    y = np.random.randint(2, size=num_samples)
    
    # Convert to DataFrame for easier handling
    feature_names = [f"feature_{i}" for i in range(num_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.Series(y, name='target')
    
    return X_df, y_df

def print_artificial_dataset(test_accuracy, tree_rules, num_leaf_nodes, optimal_max_depth):
    print(f"Artificial Dataset Test Accuracy: {test_accuracy}")
    print(f"Optimal Max Depth for Artificial Dataset: {optimal_max_depth}")
    print("If-Then Rules for Artificial Dataset:\n", tree_rules)
    print(f"Number of If-Then Clauses for Artificial Dataset: {num_leaf_nodes}")


X_artificial, y_artificial = create_artificial_dataset(num_samples=800, num_features=10)
test_accuracy, tree_rules, num_leaf_nodes, optimal_max_depth = train_and_evaluate(X_artificial, y_artificial)
print_artificial_dataset(test_accuracy, tree_rules, num_leaf_nodes, optimal_max_depth)

X_artificial, y_artificial = create_artificial_dataset(num_samples=200, num_features=3)
test_accuracy, tree_rules, num_leaf_nodes, optimal_max_depth = train_and_evaluate(X_artificial, y_artificial)
print_artificial_dataset(test_accuracy, tree_rules, num_leaf_nodes, optimal_max_depth)

X_artificial, y_artificial = create_artificial_dataset(num_samples=600, num_features=12)
test_accuracy, tree_rules, num_leaf_nodes, optimal_max_depth = train_and_evaluate(X_artificial, y_artificial)
print_artificial_dataset(test_accuracy, tree_rules, num_leaf_nodes, optimal_max_depth)

