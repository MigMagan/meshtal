#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:56:18 2021
Test for the tracer and sampling modules
@author: mmagan
"""
from math import pi
import numpy as np
import meshtal
import ptrac

def refro():
    " Reference density matrix from analytical calculation"
    ro = np.zeros((5, 5, 5))
    # Central cube
    ro[1:4, 1:4, 1:4]+=0.125*2.3
    ro[2, 1:4, 1:4]*=2
    ro[1:4, 2, 1:4]*=2
    ro[1:4, 1:4, 2]*=2
# Metal FeCu balls at +X
    ro[4, 2, 2]+=(4/3*pi*5**3/1000*7.8)
    for j in range(5):
        for k in range(5):
            if j != 2 and k !=2:
                ro[4, j, k]+=5**3/3*pi/1000*7.8
# methalic Cylinders at +Y
    ro[2, 4, 2]+= pi/10*9
    for i in range(5):
        for k in range(5):
            if i !=2 and k!=2:
                ro[i, 4, k]+=pi/40*9
# Upper porous cubes
    ro[2, 2, 4]+=0.64
    for i in (0, 4):
        for j in (0, 4):
            ro[i, j, 4]+=0.25
    return ro

def report(r):
    """compare r to the reference result and print some reports"""
    rr = refro()
    diff = r -rr
    if r.nonzero()[0].size != rr.nonzero()[0].size:
        print("Not the same number of voids. BOOOOOH!!")
    elif [all(r.nonzero()[i] == rr.nonzero()[i]) for i in range(3)] == [True, True, True]:  # Yeah, this sucks
        print("All voids match. YAY!!!")
    else:
        print("Voids do not match. BOOOOOH!!")
    meandiff = diff.mean()
    meddiff = np.median(diff)
    maxdiff = diff.max()
    stddiff = np.std(diff)
    print(" Maximum divergence: {0:5.4g}".format(maxdiff))
    print(" Median divergence should be 0, it is: {0:5.4g}".format(meddiff))
    print(" Average divergence should be very close to 0. Value: {0:5.4g}".format(meandiff))
    print(" Std of divergence: {0:5.4g}".format(stddiff))


def test_raytrace():
    """" Test the raytracing density calculation againts the analytic result
    , and plot some statistics"""


    m = meshtal.fget(4)
    ro = -ptrac.romesh(m, "ptrac_ray_bin", "outp", method="raytrace", pformat="bin")[0]
    report(ro)


def test_pointsample():
    """" Test the point sampling density calculation againts the analytic result
    , and plot some statistics"""

    m = meshtal.fget(4)
    ro = -ptrac.romesh(m, "ptrac_src_bin", "outp", pformat="bin")[0]
    report(ro)
