def polynomial(point):
    return lambda x: sum([x**i * point[i] for i in range(len(point))])