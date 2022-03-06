"""
user: doguilmak

Physical Geodesy
Computing Normalized Associated Legendre Functions

"""


def legd(nmax=None, x=None):
    import numpy as np
    import math as m
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    nmx = nmax + 1
    t = x * m.pi / 180
    p = np.zeros((nmx, nmx))
    p[0, 0] = 1
    p[1, 1] = m.sqrt(3) * m.sin(t)

    if nmx >= 2:
        for f in range(3, nmx + 1):
            p[f - 1, f - 1] = m.sqrt((2 * (f - 1) + 1) / (2 * (f - 1))) * m.sin(t) * p[f - 2, f - 2]

    for k in range(0, nmx):
        if k == 1:
            p[k, k - 1] = m.sqrt(3) * m.cos(t)

        elif k == 2:
            p[k, k - 1] = m.sqrt(5) * m.cos(t) * p[k - 1, k - 1]

        else:
            p[k, k - 1] = m.sqrt(2 * k + 1) * m.cos(t) * p[k - 1, k - 1]
            p[0, nmax] = 0

        for g in range((k + 1), nmx):
            p[g, k] = m.sqrt(((2 * g - 1) * (2 * g + 1)) / ((g - k) * (g + k))) * p[g - 1, k] * m.cos(t) - m.sqrt(
                ((2 * g + 1) * (g + k - 1) * (g - k - 1)) / ((g - k) * (g + k) * (2 * g - 3))) * p[g - 2, k]

    print("Pnm Matrix:\n")

    df = pd.DataFrame(p, columns=np.arange(0, nmx), index=np.arange(0, nmx))
    df.index.name = 'n'
    df.columns.name = 'm'
    pd.set_option('display.max_rows', None)
    pd.set_option('max_columns', None)
    print(df)

    """
    # Writing results of the calculation on .txt file:
    np.savetxt("legd_Output.txt", p, fmt="%s")
    """

    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 8))
    plt.title("Computing Normalized Associated Legendre Functions", fontsize=16)
    plt.xlabel("Number of n")
    plt.plot(p)
    plt.show()


legd(7, 45)
