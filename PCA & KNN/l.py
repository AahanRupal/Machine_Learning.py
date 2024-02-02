

K = 5
actual_predicted = []

for k in range(len(projection_test)):
    distance = []
    for z in range(len(projection_train)):
        distance_ = np.sqrt(np.sum((projection_test[k] - projection_train[z])**2))
        distance.append([distance_, y_train[z]])

    distance.sort()
    nearest_distance = distance[:K]  # Select the K-nearest neighbors

    predicted_class = Counter(top_K_neighbor[1] for top_K_neighbor in nearest_distance).most_common(1)[0][0]
    actual_predicted.append((y_test[k], predicted_class))
    
matrix = np.array(actual_predicted)
print(matrix)


for k in range(len(projection_test)):
    distance=[]
    for z in range(len(projection_train)):
        distance_=np.sqrt(np.sum((projection_test[k]-projection_train[z])**2))
        distance.append([distance_,y_train[z]])

    distance.sort()
    nearest_distance=distance[:k]
    
    predicted_class = Counter(top_K_neighbor[1] for top_K_neighbor in nearest_distance).most_common(1)[0][0]
    actual_predicted.append((y_test[k],predicted_class))
