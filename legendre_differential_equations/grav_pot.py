"""
user: doguilmak

Calculating Gravity Potential
"""


def grav_pot(lon=None, lat=None):
    import math as m
    GM = 3986005E8  # m3/s2
    omega = 7.292115E-5  # s-1
    R = 6371000  # m

    V = GM / R  # Gravity potential
    phi = 1 / 2 * omega ** 2 * (R * m.sin(m.radians(lat))) ** 2  # Centrifugal potential
    W = V + phi  # Gravity potential

    print("Gravity potential:", V, "m2/s2")
    print("Centrifugal potential:", phi, "m2/s2")
    print("Gravity potential:", W, "m2/s2")


grav_pot(0, 52)
