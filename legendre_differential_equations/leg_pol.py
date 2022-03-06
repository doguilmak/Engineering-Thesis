"""

user: doguilmak
Physical Geodesy Legendre Function

"""


def leg_pol(rang):
    from scipy.special import legendre
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    min_value = -1.0
    max_value = 1.0
    step = 0.03

    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 8))

    for n in range(rang):
        pn = legendre(n)
        x = np.arange(min_value, max_value + step, step)
        y = pn(x)
        plt.plot(x, y, label='n=' + str(n))

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title('Legendre Functions(Polynomial)', fontsize=16)
    plt.legend()
    plt.show()


leg_pol(5)
