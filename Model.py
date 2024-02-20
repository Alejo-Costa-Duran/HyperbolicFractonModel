import Geodesics as gd
import hypertiling as ht
import matplotlib.pyplot as plt
import cmath as cmt
import itertools
import numpy as np
import more_itertools as mit

class FractonModel:
    def __init__(self, p: int, q: int, nlayers : int,kernel = None):
        if((p-2)*(q-2)<=4):
            raise ValueError
        else:
            if kernel == None:
                self.lattice = ht.HyperbolicTiling(p,q,nlayers)
            else:
                self.lattice = ht.HyperbolicTiling(p,q,nlayers, kernel = kernel)
            self.bulk = [index for index in range(len(self.lattice)) if self.lattice.get_layer(index) !=nlayers]
            self.border = [index for index in range(len(self.lattice)) if self.lattice.get_layer(index) ==nlayers]
            self.borderPhases = [cmt.phase(self.lattice.get_center(i)) for i in self.border]
            self.borderPhases, self.border = mit.sort_together([self.borderPhases, self.border])
            self.border = list(self.border)
            self.geodesicList = self.getGeodesics()
            self.spins = np.array([1 for _ in range(len(self.lattice))])
            self.borderNeigh = [[(idx +j)%len(self.border) for j in range(1,int(len(self.border)/2)+1)] for idx in range(len(self.border))]
            self.centers = np.array([self.lattice.get_center(idx) for idx in range(len(self.lattice))])
            self.interactions = self.getInteractionMatrix()

    def getBorderCorrelations(self):
        borderSpins = self.spins[self.border]
        return np.sum(borderSpins[self.borderNeigh]*borderSpins[:,None],axis = 0)

    def hamiltonian(self):
        ener = 0
        for inter in self.interactions:
            prod = 1
            for vec in inter:
                prod *= self.spins[vec]
            ener += prod
        return -ener
    
    def getInteractionMatrix(self):
        nPol = len(self.lattice)
        vertex_list = []
        int_list = []
        for pol in self.bulk:
            vertices = self.lattice.get_vertices(pol)
            for i in range(len(vertices)):
                if vertices[i] not in vertex_list:
                    vertex_list.append(vertices[i])
                    temp = []
                    for pol1 in range(nPol):
                        vertices1 = self.lattice.get_vertices(pol1)
                        for v in vertices1:
                            if cmt.isclose(v,vertices[i]):
                                temp.append(pol1)
                    int_list.append(temp)
        return int_list

    def getGeodesics(self):
        geodesic_list = []
        for pol_index in self.bulk:
            for vertex in range(self.lattice.p): 
                g_new = gd.Geodesics(self.lattice.get_vertices(pol_index)[vertex],self.lattice.get_vertices(pol_index)[(vertex+1)%self.lattice.p])
                add = True
                for g in geodesic_list:
                    if g==g_new:
                        add = False
                        break
                if add:
                    geodesic_list.append(g_new)
        return geodesic_list


    def quick_plot(self, unitcircle=False, fig = None, ax = None,dpi=150,colors = None):
        if colors == None:
            colors = ['w' for _ in range(len(self.lattice))]
        if len(colors) == 1:
            fill_color = colors[0]
            colors = [fill_color for _ in range(len(self.lattice))]
        if colors == 'spins':
            colors = []
            for index in range(len(self.lattice)):
                if self.spins[index] == 1:
                    colors.append('b')
                elif self.spins[index] == -1:
                    colors.append('r')
                else:
                    colors.append('w')
        # actual plot
        if fig == None and ax==None:
            fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.axis('equal')
            plt.axis('off')
        
        # add bounding circle
        if unitcircle:
            circle = plt.Circle((0, 0), 1)
            ax.add_patch(circle)

        # draw polygons
        x, y = [], []
        for i in range(len(self.lattice)):
            v = self.lattice.get_vertices(i)
            v = np.append(v, v[0])  # appending first vertex to close the path
            x.extend(v.real)
            x.append(None)  # this is some kind of trick that makes it that fast
            y.extend(v.imag)
            y.append(None)
        polygons = [list(group) for key, group in itertools.groupby(zip(x, y), lambda x: x[0] is None) if not key]

        for i, polygon in enumerate(polygons):
            x0, y0 = zip(*polygon)
            ax.fill(x0, y0, color=colors[i],edgecolor ='k')
        return (fig,ax)