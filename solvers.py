#We solve the system with boundary conditions directly

def Laplace2Dsolver(N, f):
    
    #Makes the matrix without considering the boundary
    Atilde =  StiffnessMatrix(N)
    
    #Makes the right hand side without considering the boundary
    Ftilde = LoadVector(N, f)
    
    p,tri,edge = GetDisc(N)
    
    #Define the size the reduced system will have, N - #boundary-nodes
    intN = N - np.size(edge,0)
    
    #Slice the matrix to impose the boundary
    A = Atilde[:intN, :intN]
    
    #Slice the F-vector to be correct size
    F = Ftilde[:intN]
    
    #Solve the system
    u = np.linalg.solve(A,F)
    
    return u


def isNeumann(idx):
     return p[idx][1]>0 #here one can choose if >=0 or >0


def Laplace2DsolverNeumann(N, f, g):
    Atilde =  StiffnessMatrix(N)
    Ftilde = LoadVectorNeumann(N, f, g)
    
    p,tri,edge = GetDisc(N)
    
    #Makes the matrix without considering the boundary
    Atilde =  StiffnessMatrix(N)
    
    #Makes the right hand side without considering the boundary
    Ftilde = LoadVector(N, f)
    
    p,tri,edge = GetDisc(N)
    
    #Define the size the reduced system will have, N - #boundary-nodes
    intN = N - np.size(edge,0)
    
    #Slice the matrix to impose the boundary
    A = Atilde[:intN, :intN]
    
    #Slice the F-vector to be correct size
    F = Ftilde[:intN]
    
    
    res = np.zeros(N)
    
    for e in edge:

        idx_1 = e[0]

        idx_2 = e[1]

        if isNeumann(idx_1) or isNeumann(idx_2):

            if isNeumann(idx_1):

                res[idx_1] += linequadrature2D(p[idx_1],p[idx_2],3,g)

            if isNeumann(idx_2):

                res[idx_2] += linequadrature2D(p[idx_1],p[idx_2],3,g)


    F = F + res[:intN]

    
    #Solve the system
    u = np.linalg.solve(A,F)
    
    return u
