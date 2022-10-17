def LoadVector(N, f):
    # N is number of nodes in the triangulation
    # f is the right hand side
    
    p,tri,edge = GetDisc(N)
    
    #Solution storrage
    F = np.zeros(N)
    
    # Loop over all elements
    for k in range(len(tri)): 
        
        #Finds the coefficients of the basis functions for the current element
        C = np.linalg.solve([[1,p[tri[k,0],0],p[tri[k,0],1]],
                             [1,p[tri[k,1],0],p[tri[k,1],1]],
                             [1,p[tri[k,2],0],p[tri[k,2],1]]],
                            np.identity(3))
        # Loop over the corners of the current triangle
        for alpha in range(3):
            i = tri[k, alpha] #Lokal to global map
            
            #Makes the basis function
            def H_ak(x,y): 
                return C[0,alpha] + C[1,alpha]*x + C[2,alpha]*y
            
            #Makes the function inside the integral
            def fH_ak(x,y):
                return f(x,y)*H_ak(x,y)
            
            g = lambda x,y: fH_ak(x,y)
            
            #Fill inn the vector by solving the integral with quadrature.
            F[i] = F[i] + quadrature2D(p[tri[k,0]], p[tri[k,1]], p[tri[k,2]], 3, g)
    return F

def LoadVectorNeumann(N, f, g):
    
    p,tri,edge = GetDisc(N)
    
    F = np.zeros(N)
    
    Nboundary = []
    for i in range(N):
        if i in edge[:,0] and p[i,1] >= 0:
            Nboundary.append(i) 
    
    for k in range(len(tri)):
        C = np.linalg.solve([[1,p[tri[k,0],0],p[tri[k,0],1]],
                             [1,p[tri[k,1],0],p[tri[k,1],1]],
                             [1,p[tri[k,2],0],p[tri[k,2],1]]],
                            np.identity(3))
        for alpha in range(3):
            i = tri[k, alpha]
            
            def H_ak(x,y):
                return C[0,alpha] + C[1,alpha]*x + C[2,alpha]*y
            def fH_ak(x,y):
                return f(x,y)*H_ak(x,y)  
            
            if i in Nboundary:
                nodes = []
                for j in range(3):
                    if tri[k,j] in Nboundary:
                        nodes.append(tri[k,j])
                if len(nodes) == 2:
                    def gH_ak(x,y):
                        return g(x,y)*H_ak(x,y)
                    nodes.sort()
                    F[i] = F[i] + linequadrature2D(p[nodes[0],:],p[nodes[1],:],4,gH_ak)
            
            
            h = lambda x,y: fH_ak(x,y)
            
            F[i] = F[i] + quadrature2D(p[tri[k,0]], p[tri[k,1]], p[tri[k,2]], 3, h)
    return F 
