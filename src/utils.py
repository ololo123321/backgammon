def rolls_gen():
    for i in range(1, 7):
        for j in range(i, 7):
            if i == j:
                yield (i,) * 4
            else:
                yield i, j
