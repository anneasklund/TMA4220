from getdisc import GetDisc, NodalPoints, FreeBoundary, CircleData
import matrices
import loadvectors
import quadrature


#Right hand side in the model problem
def f(x,y):
    return -8*np.pi*np.cos(2*np.pi*(x**2+y**2)) + 16*np.pi**2*(x**2 + y**2)*np.sin(2*np.pi*(x**2 + y**2))

#Exact solution of the model problem
def u_ex(x,y):
    return np.sin(2*np.pi*(x**2 + y**2))

N = 3000

p,tri,edge = GetDisc(N)
u = Laplace2Dsolver(N, f)
u_w_boundary = np.zeros(len(p))
u_w_boundary[:len(u)] = u
    
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_trisurf(p[:,0],p[:,1],u_w_boundary)
ax.set_title('Numeric solution')
#fig.savefig('2_8_numeric.pdf')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_trisurf(p[:,0],p[:,1],u_ex(p[:,0],p[:,1]))
ax.set_title('Exact solution')
#fig.savefig('2_8_exact.pdf')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_trisurf(p[:,0],p[:,1],u_ex(p[:,0],p[:,1])-u_w_boundary)
ax.set_title('Error')
#fig.savefig('2_8_error.pdf')


def g(x,y):
    return 4*np.pi*np.sqrt(x**2+y**2)*np.cos(2*np.pi*(x**2+y**2)) 

N = 2001

p,tri,edge = GetDisc(N)
u = Laplace2DsolverNeumann(N, f, g)
u_w_boundary = np.zeros(len(p))
u_w_boundary[:len(u)] = u  
    
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#ax.plot_trisurf(p[:len(u),0],p[:len(u),1],u)
ax.plot_trisurf(p[:,0],p[:,1],u_w_boundary)
ax.set_title('Numeric solution')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_trisurf(p[:,0],p[:,1],u_ex(p[:,0],p[:,1])-u_w_boundary)
ax.set_title('Error')

fig.show()
