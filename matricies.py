def StiffnessMatrix(N):
    # Function which creates the stiffness matrix
    # for a triangulation with N nodes
    
    # Creates function g = 1 to use quadrature to 
    # find the area of the triangles
    g = lambda x,y: 1
    
    # Get the triangulation
    p,tri,edge = GetDisc(N)
    
    A = np.zeros((N,N))
    
    # Loop over the elements to build A
    for k in range(len(tri)):
        area = quadrature2D(p[tri[k,0]], p[tri[k,1]], p[tri[k,2]], 3, g)
        
        # Find coefficients of the basis functions in element k
        C = np.linalg.solve([[1,p[tri[k,0],0],p[tri[k,0],1]],
                             [1,p[tri[k,1],0],p[tri[k,1],1]],
                             [1,p[tri[k,2],0],p[tri[k,2],1]]],
                            np.identity(3))
        
        for alpha in range(3):
            i = tri[k,alpha]
            for beta in range(3):
                j = tri[k,beta]
                A[i,j] = A[i,j] + area * (C[1,alpha] * C[1,beta] + C[2,alpha] * C[2,beta])
    return A

