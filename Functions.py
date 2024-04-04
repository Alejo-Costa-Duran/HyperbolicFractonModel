def isBetween(z0,z1,z2):
    cross10 = z1.real*z0.imag-z0.real*z1.imag
    cross12 = z1.real*z2.imag-z2.real*z1.imag
    cross20 = z2.real*z0.imag-z0.real*z2.imag
    cross21 = -z1.real*z2.imag+z2.real*z1.imag
    isBe = (cross10*cross12) >= 0 and (cross20*cross21) >= 0
    return isBe

def insidePolygons(model):
    insides_list = []
    idx= 0
    for geod in model.geodesicList:
        inside_g = geod.vect_inside()(model.centers[model.border])
        insides_list.append(inside_g)
        idx+=1
    return insides_list

def entanglement_entropy(model, blackHoleRad, freq = 1):
    borderCenters = model.centers[model.border]
    borderNeigh = np.array(borderCenters[model.borderNeigh])
    wedges = np.zeros(len(borderCenters)-1)
    validGeodesics =[g for g in model.geodesicList if g.distanceToOrigin > blackHoleRad]
    for idx,z1 in enumerate(borderCenters):
        if idx%freq==0:
            start = time.time()
            for idx2,z2 in enumerate(borderNeigh[idx,:]):
                for g in validGeodesics:
                    x = isBetween(g.endpoints[0],z1,z2) 
                    y = isBetween(g.endpoints[1],z1,z2)
                    if bool((x and not y) or (not x and y)):
                        wedges[idx2] += 1
                        wedges[len(wedges)-idx2-1] += 1
            end = time.time()
            print('Computing entropy. Time remaining: ' + str(round((end-start)*(len(borderCenters)/freq-idx/freq)/60,4))+' m'+' '*100, end = '\r')
    wedges[int((len(wedges)-1)/2)] = wedges[int((len(wedges)-1)/2)]/2
    return np.array(wedges)/len(model.border)

def geodesicsePerLength(modl):
    corr = np.zeros(len(modl.border))
    borderCenters = modl.centers[modl.border]
    for idx2,g in enumerate(modl.geodesicList):
        func = g.vect_inside()
        nums = func(borderCenters)
        idx = int(-np.sum(nums)+len(borderCenters))
        corr[int(idx/2)-1] +=1
    return corr