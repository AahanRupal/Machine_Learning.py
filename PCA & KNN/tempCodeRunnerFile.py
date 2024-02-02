predicted_class = Counter(top_K_neighbor[1] for top_K_neighbor in nearest_distance).most_common(1)[0][0]
    