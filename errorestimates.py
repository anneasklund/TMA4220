from getdisc import GetDisc, NodalPoints, FreeBoundary, CircleData
import matrices
import quadrature
import solvers

def LoadVectorNeumann(N, f, g):

    p,tri,edge = GetDisc(N)

    F = np.zeros(N)

    res = np.zeros(N)

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

            h = lambda x,y: fH_ak(x,y)

            F[i] = F[i] + quadrature2D(p[tri[k,0]], p[tri[k,1]], p[tri[k,2]], 3, h)

    for e in edge:

        idx_1 = e[0]

        idx_2 = e[1]

        if isNeumann(idx_1) or isNeumann(idx_2):

            if isNeumann(idx_1):

                res[idx_1] += linequadrature2D(p[idx_1],p[idx_2],3,g)/2

            if isNeumann(idx_2):
                res[idx_2] += linequadrature2D(p[idx_1],p[idx_2],3,g)/2


    F = F + res

    return F

def isNeumann(idx):
     return p[idx][1]>=0 #here one can choose if >=0 or >0
    def Laplace2DsolverNeumann(N, f, g):
    Atilde =  StiffnessMatrix(N)
    Ftilde = LoadVectorNeumann(N, f, g)

    p,tri,edge = GetDisc(N)

    #Makes the matrix without considering the boundary
    Atilde =  StiffnessMatrix(N)

    #Makes the right hand side without considering the boundary
    Ftilde = LoadVectorNeumann(N, f, g)

    p,tri,edge = GetDisc(N)

    NDboundary = 0
    for i in range(N):
        if i in edge[:,0] and p[i,1] < 0:
            NDboundary += 1
    Nend = N - NDboundary

    #Define the size the reduced system will have, N - #boundary-nodes
    #print(len(Ftilde[Ftilde!=0]))
    #Nend = int(N-np.floor(len(edge)/2)-1)
    #print(Nend)
    #Slice the matrix to impose the boundary
    A = Atilde[:Nend, :Nend]

    #Slice the F-vector to be correct size
    F = Ftilde[:Nend]

    #F = F + res[:Nend]


    #Solve the system
    u = np.linalg.solve(A,F)

    return u

#Right hand side in the model problem
def f(x,y):
    return -8*np.pi*np.cos(2*np.pi*(x**2+y**2)) + 16*np.pi**2*(x**2 + y**2)*np.sin(2*np.pi*(x**2 + y**2))

#Exact solution of the model problem
def u_ex(x,y):
    return np.sin(2*np.pi*(x**2 + y**2))

def u_ex_grad(x,y):
    return 4*np.pi*np.cos(2*np.pi*(x**2+y**2))*np.array([x,y])

def g(x,y):
    return 4*np.pi*np.sqrt(x**2+y**2)*np.cos(2*np.pi*(x**2+y**2))

N = np.array([10,100,1000,5000])

error_dirichlet = np.zeros(len(N))
error_neumann = np.zeros(len(N))

def errorestimate(p,tri,u,uh):

    error = 0

    for k in range(len(tri)):
        C = np.linalg.solve([[1,p[tri[k,0],0],p[tri[k,0],1]],
                             [1,p[tri[k,1],0],p[tri[k,1],1]],
                             [1,p[tri[k,2],0],p[tri[k,2],1]]],
                              np.identity(3))
        graduh =  uh[tri[k,0]]*np.array([C[1,0],C[2,0]])+uh[tri[k,1]]*np.array([C[1,1],C[2,1]])+uh[tri[k,2]]*np.array([C[1,2],C[2,2]])
        def diff2(x,y):
            J = u_ex_grad(x,y).T-graduh
            return np.sum(J**2,1)
        error = error + quadrature2D(p[tri[k,0]],p[tri[k,1]],p[tri[k,2]],4,diff2)

    return np.sqrt(error)

for i in range(len(N)):

        p,tri,edge = GetDisc(N[i])

        u_dirichlet = Laplace2Dsolver(N[i], f)
        u_w_boundary_dirichlet = np.zeros(len(p))
        u_w_boundary_dirichlet[:len(u_dirichlet)] = u_dirichlet
        error_dirichlet[i] = errorestimate(p,tri,u_ex,u_w_boundary_dirichlet)

        u_neumann = Laplace2DsolverNeumann(N[i], f, g)
        u_w_boundary_neumann = np.zeros(len(p))
        u_w_boundary_neumann[:len(u_neumann)] = u_neumann
        error_neumann[i] = errorestimate(p,tri,u_ex,u_w_boundary_neumann)

hs = 1/N ### Må sees mer på, kan være feil

fig, ax = plt.subplots()
ax.loglog(hs,error_dirichlet,hs,error_neumann)
ax.legend(['Dirichlet','Neumann'])
ax.set_title('Convergence rate')
ax.set_xlabel('h')
ax.set_ylabel('Error')

fig.show()

