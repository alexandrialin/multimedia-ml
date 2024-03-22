import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def generate_random_points(n, D, num_classes):
    X = np.random.rand(n, D)
    y = np.random.randint(num_classes, size=n)
    return X, y

def memorization_requirement(X, y):
    knn = KNeighborsClassifier(n_neighbors=1)
    n = len(y)
    indices = list(range(n))
    essential_indices = []

    while len(indices) > 1:
        removed_index = indices.pop(np.random.randint(0, len(indices)))
        X_train, y_train = X[essential_indices + indices], y[essential_indices + indices]
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X)
        if not np.all(y_pred == y):
            essential_indices.append(removed_index)

    if indices and indices[0] not in essential_indices:
        essential_indices.append(indices[0])
    return len(essential_indices)

dimensions = [2, 4, 8]
num_classes_list = [3, 4, 5] 

datasets_per_dimension = {2: 8, 4: 16, 8: 32}

for d in dimensions:
    for num_classes in num_classes_list:
        mem_requirements = []
        for _ in range(datasets_per_dimension[d]):
            X, y = generate_random_points(2 ** d, d, num_classes)
            mem_req = memorization_requirement(X, y)
            mem_requirements.append(mem_req)

        avg_mem_req = np.mean(mem_requirements)
        print(f"d={d}, k={num_classes}: n_full={2 ** d}, "
              f"Avg. req. points for memorization n_avg={avg_mem_req:.2f}, "
              f"n_full/n_avg={(2 ** d)/avg_mem_req:.2f}")

