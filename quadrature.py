def quadrature1D(a, b, Nq, g):
    # Function which performs quadrature in 1D
    # of function g over the interval [a,b] with
    # Nq quadrature points where Nq = 1, 2, 3, or 4

    # Calculating half of the interval length
    h = (b-a)/2
    
    # Setting quadrature points and weights from table (barycentric coordinates)
    if Nq == 1:
        zq = 0
        pq = 2
        I = h * (pq * g(h*zq+(a+b)/2))
    elif Nq == 2:
        zq = np.array([-np.sqrt(1/3),np.sqrt(1/3)])
        pq = np.array([1,1])
    elif Nq == 3:
        zq = np.array([-np.sqrt(3/5),0,np.sqrt(3/5)])
        pq = np.array([5/9,8/9,5/9])
    elif Nq == 4:
        zq = np.array([-np.sqrt((3+2*np.sqrt(6/5))/7),-np.sqrt((3-2*np.sqrt(6/5))/7),
                       np.sqrt((3-2*np.sqrt(6/5))/7),np.sqrt((3+2*np.sqrt(6/5))/7)])
        pq = np.array([(18-np.sqrt(30)),(18+np.sqrt(30)),(18+np.sqrt(30)),(18-np.sqrt(30))])/36
    else:
        print("Error: value of Nq is not allowed")
        
    # Calculating the approx integral I
    if Nq != 1:
        I = h * sum(pq * g(h*zq+(a+b)/2))

    return I

def quadrature2D(p1, p2, p3, Nq, g):
    # Function which performs quadrature in 2D
    # of function g over the triangle T with verticies
    # p1, p2, and p3 with Nq quadrature points where 
    # Nq = 1, 3, or 4
    
    # Jacobian of T
    J = np.linalg.norm(np.cross(p1-p3,p2-p3))
    
    # Setting quadrature points and weights from table (barycentric coordinates)
    if Nq == 1:
        zq = np.ones(3)/3
        pq = 1
        coord = zq[0]*p1 + zq[1]*p2 + zq[2]*p3
        I = J/2 * pq * g(coord[0],coord[1])
    elif Nq == 3:
        zq = np.array([[1/2,1/2,0],[1/2,0,1/2],[0,1/2,1/2]])
        pq = np.ones(3)/3
    elif Nq == 4:
        zq = np.array([[1/3,1/3,1/3],[3/5,1/5,1/5],[1/5,3/5,1/5],[1/5,1/5,3/5]])
        pq = np.array([-9/16,25/48,25/48,25/48])
    else:
        print("Error: value of Nq is not allowed")
        
    # Calculating the coordinates in real-space and approx integral I
    if Nq != 1:
        coords = np.outer(zq[:,0],p1) + np.outer(zq[:,1],p2) + np.outer(zq[:,2],p3 )
        I = J/2 * sum(pq * g(coords[:,0],coords[:,1]))
    
    return I

def linequadrature2D(a,b,Nq,g):
    #Takes two points a and b in 2D, Nq which is the number of interpolation points, and a function g. 
    #Returns the straight-line integral from a to b of g. 
    
    #These quanteties are discribed in the report. 
    norm = (1/2)*np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)
    
    def x(t): 
        return (1/2)*((1-t)*a[0] + (t+1)*b[0])
    def y(t): 
        return (1/2)*((1-t)*a[1] + (t+1)*b[1])
    def f(t):
        return g(x(t), y(t))
    
    u = lambda t: f(t) 
    
    return norm*quadrature1D(-1, 1, Nq, u)
