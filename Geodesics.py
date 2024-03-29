import math as mt
import cmath as cmt
import numpy as np

class Geodesics:
    """
    Geodesics class. Wrapper to hold the parameters of the polygon geodesics on a hyperbolic lattice.
    """
    def __init__(self,z1: complex,z2: complex):
        self.center = self.computeCenter(z1,z2)
        self.radius = abs(self.center)**2-1
        self.endpoints = self.generateEndpoints()
        self.distanceToOrigin = self.computeDistance()


    def computeCenter(self,z1: complex,z2: complex):
        """
            Input: z1,z2 complex numbers.
            Output: Center of the geodesic that passes through z1 and z2.
        """
        a = (z1.imag*(abs(z2)**2+1)-z2.imag*(abs(z1)**2+1))/(z1.real*z2.imag-z2.real*z1.imag)
        b = (z2.real*(abs(z1)**2+1)-z1.real*(abs(z2)**2+1))/(z1.real*z2.imag-z2.real*z1.imag)
        return a/2+1j*b/2
        
    
    def computeDistance(self):
        """
            Output: Euclidean distance from the origin to the geodesic in the Poincare disk.
        """
        a = 2*self.center.real
        b = 2*self.center.imag
        if b==0:
            return min(abs(mt.sqrt(-1+a*a/4)-a/2),abs(-mt.sqrt(-1+a*a/4)-a/2))
        if a==0:
            return min(abs(mt.sqrt(-1+b*b/4)-b/2),abs(-mt.sqrt(-1+b*b/4)-b/2))
        else:
            u = mt.sqrt(mt.pow(a,6)+2*mt.pow(a,4)*(b*b-2)+a*a*b*b*(b*b-4))
            x0 = -(mt.pow(a,3)+u+a*b*b)/(2*(a*a+b*b))
            y0 = -b*(mt.pow(a,3)+u+a*b*b)/(2*a*(a*a+b*b))
            x1 = -(mt.pow(a,3)-u+a*b*b)/(2*(a*a+b*b))
            y1 = -b*(mt.pow(a,3)-u+a*b*b)/(2*a*(a*a+b*b))
            return min(mt.sqrt(x0*x0+y0*y0),mt.sqrt(x1*x1+y1*y1))

    def isInside(self,z0: complex) -> bool:
        """
            Input: z0, complex number in the Poincaré Disk, not in the geodesic.
            Output: Returns 1 if z0 is on one side of the geodesic, -1 if its on the other side.
        """
        z1 = self.endpoints[0]
        z2 = self.endpoints[1]
        cross10 = z1.real*z0.imag-z0.real*z1.imag
        cross12 = z1.real*z2.imag-z2.real*z1.imag
        cross20 = z2.real*z0.imag-z0.real*z2.imag
        cross21 = -z1.real*z2.imag+z2.real*z1.imag
        isBetween = (cross10*cross12) >= 0 and (cross20*cross21) >= 0
        isOuter = abs(z0) > self.distanceToOrigin 
        if (isOuter and isBetween):
            return -1
        else:
            return 1
        
    def vect_inside(self):
        """
            Vectorizes the isInside function
        """
        return np.vectorize(self.isInside)

    def generateEndpoints(self):
        """
            Output: Points where the geodesic crosses the unit circle.
        """
        a = 2*self.center.real
        b = 2*self.center.imag
        if a == 0:
            y0 = -2/b
            x01 = mt.sqrt(1-4/b**2)
            x02 = -mt.sqrt(1-4/b**2)
            return [x01+1j*y0,x02+y0*1j]
        elif b == 0:
            x0 = -2/a
            y01 = mt.sqrt(1-4/a**2)
            y02 = -mt.sqrt(1-4/a**2)
            return [x0+1j*y01,x0+1j*y02]
        else:
            x0 = -(mt.sqrt(b*b*(a*a+b*b-4))+2*a)/(a*a+b*b)
            y0 = (a*mt.sqrt(b*b*(a*a+b*b-4))-2*b*b)/(b*(a*a+b*b))
            x01 = (mt.sqrt(b*b*(a*a+b*b-4))-2*a)/(a*a+b*b)
            y01 = -(a*mt.sqrt(b*b*(a*a+b*b-4))+2*b*b)/(b*(a*a+b*b))
            return [x0+y0*1j,x01+y01*1j]

    def __eq__(self, other):
        """
            Checks if two geodesics are equal
        """
        return np.abs(self.center-other.center) < 0.0000001