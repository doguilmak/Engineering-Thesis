# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 01:10:16 2021

@author: doguilmak

Calculating Gravitational Potential, Centrifugal Potential and Gravity Potential on Every Latitude
"""


def calc_grav(lat=None):
    import math as m
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    e = 2 * (1 / 298.257222101) - (1 / 298.257222101) ** 2
    GM = 3986005E8  # m³/s²
    omega = 7.292115E-5  # s-1

    v_array = []
    phi_array = []
    w_array = []
    radius_array = []
    latitude = np.arange(0, lat + 0.01, 0.1)

    try:
        for i in np.arange(0, lat + 0.01, 0.1):
            radius = 6378137 / m.sqrt(1 - e * m.sin(i * (m.pi / 180)) ** 2)  # Transverse radius of curvature
            v = GM / radius  # Gravity potential
            phi = 1 / 2 * omega ** 2 * (radius * m.sin(m.radians(i))) ** 2  # Centrifugal potential
            W = v + phi  # Gravitational potential

            # Saving output into .txt file:
            """
            print("Latitude", i, file=open("output.txt", "a"))
            print("Radius of the earth in latitude of {} is {} meters".format(i, radius), file=open("output.txt", "a"))
            print("Gravity potential:", v, "m²/s²", file=open("output.txt", "a"))
            print("Centrifugal potential:", phi, "m²/s²", file=open("output.txt", "a"))
            print("Gravitational potential:", W, "m²/s²", file=open("output.txt", "a"))
            print("\n", file=open("output.txt", "a"))
            """

            print("Latitude", i)
            print("Radius of the earth in latitude of {} is {} meters".format(i, radius))
            print("Gravity potential:", v, "m²/s²")
            print("Centrifugal potential:", phi, "m²/s²")
            print("Gravitational potential:", W, "m²/s²")
            print("\n")

            radius_array.append(radius)
            v_array.append(v)
            phi_array.append(phi)
            w_array.append(W)

    except ValueError:
        print("Wrong input.")

    finally:
        sns.set_style("whitegrid")
        plt.figure(figsize=(18, 16))
        plt.subplot(3, 3, (1, 3))
        plt.plot(latitude, radius_array, c='red')
        plt.xlabel("Latitude")
        plt.ylabel("Radius of the earth as m")
        plt.title("Calculating Gravitational Potential, Centrifugal Potential and Gravity Potential on Every Latitude",
                  fontsize=14)

        plt.subplot(3, 2, 3)
        plt.plot(latitude, phi_array)
        plt.xlabel("Latitude")
        plt.ylabel("Centrifugal potential as m²/s²")
        plt.ylim(6.249482e+07, -6.249482e+07)

        plt.subplot(3, 2, 4)
        plt.plot(latitude, phi_array, color="orange")
        plt.xlabel("Latitude")
        plt.ylabel("Centrifugal potential as m²/s²")
        plt.xlim(50, 130)

        plt.subplot(3, 2, 5)
        plt.plot(latitude, v_array, c='black')
        plt.xlabel("Latitude")
        plt.ylabel("Gravity potential as m²/s²")
        plt.xlim(50, 130)
        plt.ylim(6.2278E7, 6.25E7)

        plt.subplot(3, 2, 6)
        plt.plot(latitude, w_array, c='green')
        plt.xlabel("Latitude")
        plt.ylabel("Gravitational potential as m²/s²")
        plt.xlim(50, 130)
        plt.ylim(6.2278E7, 6.25E7)
        plt.show()


calc_grav(180)
