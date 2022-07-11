
def kaka(mat, name='matrix_visualization.png'):
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 40))
    plt.imshow(mat, vmin=1, vmax=5, aspect="equal")
    plt.tight_layout()
    plt.savefig(name)
