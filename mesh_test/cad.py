from math import cos, sin, radians, atan
from math import atan
import numpy as np
import cadquery as cq

def naca2412(x, c=1):
    m = 0.02
    p = 0.4
    t = 0.12

    yt = (t/0.2)*c*(0.2969*np.sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)**2 + 0.2843*(x/c)**3 - 0.1036*(x/c)**4)

    if x < p*c:
        yc = (m/p**2)*(2*p*(x/c) - (x/c)**2)
        theta = atan((m/p**2)*(2*p - 2*(x/c)))
    else:
        yc = (m/(1-p)**2) * ((1 - 2*p) + 2*p*(x/c) - (x/c)**2)
        theta = atan((m/(1-p)**2)*(2*p - 2*(x/c)))

    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)

    return xu, yu, xl, yl

def generate_airfoil_edge_points(N=100, c=1):
    x = np.linspace(0, c, N)
    uppers, lowers = [], []
    for xi in x:
        xu, yu, xl, yl = naca2412(xi, c)
        uppers.append((xu, yu))
        lowers.append((xl, yl))
    return uppers, lowers

def build_airfoil_wing(N=100, chord_length_root=1, taper_ratio=0.5, aspect_ratio=9):
    span = aspect_ratio * chord_length_root
    tip_chord_length = chord_length_root * taper_ratio

    uppers, lowers = generate_airfoil_edge_points(N, chord_length_root)
    
    root_profile = uppers + list(reversed(lowers))
    tip_profile = [(x*tip_chord_length, y*tip_chord_length) for x, y in root_profile]

    wing = (cq.Workplane("XY")
                .polyline(root_profile)
                .close()
                .workplane(offset=span)
                .polyline(tip_profile)
                .close()
                .loft())
    return wing

obj = build_airfoil_wing(N=100, chord_length_root=0.5, taper_ratio=1, aspect_ratio=4)

step_file = "wing.step"
obj.val().exportStep(step_file)