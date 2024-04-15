import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def post_proc(N: np.ndarray, E: np.ndarray, u: np.ndarray, eType: str, F: np.ndarray, EA: float) -> Figure:
    """Post processing for the FEM results of a 1D bar element.

    Plot the displacement and axial force distribution along the bar element.:

    Parameters
    ----------
    N : np.ndarray
        Array of node coordinates.
    E : np.ndarray
        2D array of element nodes. Each row corresponds to an element.
    u : np.ndarray
        Array of nodal displacements returned from the FEM solver.
    eType : str
        Element type ('Lin1D' or 'Quad1D').
    F : np.ndarray
        Array of external forces.
    EA : float
        Product of Young's modulus and cross-sectional area.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing two axes. Left axis has the displacement plot, right axis the stress plot.
    """

    # number of nodes
    numN = N.size
    # number of elements
    numE = E.shape[0]
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))

    # colors
    fjBlue = (0, 73/255, 131/255)
    fjOrange = (244/255, 155/255, 0)

    # DISPLACEMENT
    maxU = np.max(np.abs(u))

    # plot elements and displacement
    if eType == "Lin1D":
        for i in range(0,numE):
            ind = E[i,:] - 1
            ax1.plot(N[ind], np.zeros(2), 'k-')
            ax1.plot(N[ind], u[ind], linewidth=3, color=fjBlue)
            ax1.plot([N[ind[0]],N[ind[0]]], [0, u[ind[0]]], 'k:') 
            ax1.plot([N[ind[1]],N[ind[1]]], [0, u[ind[1]]], 'k:')  
    elif eType == "Quad1D":
        for i in range(0,numE):
            ind = E[i,:] - 1
            le  = N[ind[0]] - N[ind[1]]
            uNe = u[ind]
            xi  = np.linspace(0,le,10)
            ue  = xi*(2*xi -le)/le**2*uNe[0] + (le - xi)*(le - 2*xi)/le**2*uNe[1] + 4*xi*(le - xi)/le**2*uNe[2]
            ax1.plot(N[ind[1]]+xi, ue, linewidth=3, color=fjBlue)
            ax1.plot(N[ind[0:2]], np.zeros(2), 'k-')
            ax1.plot([N[ind[0]],N[ind[0]]], [0, uNe[0]], 'k:')
            ax1.plot([N[ind[1]],N[ind[1]]], [0, uNe[1]], 'k:')
    else:
        raise ValueError("Unknown element type.")

    # plot nodes
    ax1.plot(N, np.zeros(N.size), 'ro')

    # plot node numbers and displacements
    for i in range(0,numN):
        ax1.text(N[i], -maxU/15, str(i+1), fontsize=9, verticalalignment='bottom', horizontalalignment='center')
        ax1.text(N[i], u[i]+maxU/30, f"{u[i]:2.4f}", fontsize=9, verticalalignment='bottom', horizontalalignment='center', rotation=90,color=fjBlue)

    # AXIAL FORCE
    # plot elements
    numNe = E.shape[1]
    for i in range(0,numE):
        ind = E[i,:] - 1
        ax2.plot(N[ind], np.zeros(numNe), 'k-')

    # find maximal axial force
    Fmax = 0
    for i in range(0,numE):
        ind = E[i,:] - 1
        xr = N[ind[1]]
        xl = N[ind[0]]
        le = xr - xl
        ue = u[ind]
        Fmax = np.max([EA*(ue[1]-ue[0])/le,Fmax])

    dF = Fmax/30

    # plot axial forces
    if eType == "Lin1D":
        for i in range(0,numE):
            ind = E[i,:] - 1
            xr = N[ind[1]]
            xl = N[ind[0]]
            le = xr - xl
            ue = u[ind]
            F  = EA*(ue[1]-ue[0])/le

            ax2.plot([xl, xr], [F, F], color=fjOrange, linewidth=3)
            ax2.plot([xl, xl], [0, F], 'k:')
            ax2.plot([xr, xr], [0, F], 'k:')
            if F<0:
                ax2.text((xr+xl)/2, F-dF, f"{F:2.2f}", fontsize=9, verticalalignment='top', horizontalalignment='center', rotation=90, color=fjOrange)
            else:
                ax2.text((xr+xl)/2, F+dF, f"{F:2.2f}", fontsize=9, verticalalignment='bottom', horizontalalignment='center', rotation=90, color=fjOrange)
    elif eType == "Quad1D":
        for i in range(0,numE):
            ind = E[i,:] - 1
            xr = N[ind[0]]
            xl = N[ind[1]]
            le = xr - xl
            ue = u[ind]
            xi = np.array([le, 0])
            Fe = EA*(-(le - 4*xi)/le**2*ue[0] - (3*le - 4*xi)/le**2*ue[1] + 4*(le - 2*xi)/le**2*ue[2])
    
            ax2.plot([xr, xl], Fe, color=fjOrange, linewidth=3)
            ax2.plot([xl, xl], [0, Fe[1]], 'k:')
            ax2.plot([xr, xr], [0, Fe[0]], 'k:')

            if Fe[0]<0:
                ax2.text(xr, Fe[0]-dF, f"{Fe[0]:2.2f}", fontsize=9, verticalalignment='top', horizontalalignment='right', rotation=90, color=fjOrange)
            else:
                ax2.text(xr, Fe[0]+dF, f"{Fe[0]:2.2f}", fontsize=9, verticalalignment='bottom', horizontalalignment='right', rotation=90, color=fjOrange)
            if Fe[1]<0:
                ax2.text(xl, Fe[1]-dF, f"{Fe[1]:2.2f}", fontsize=9, verticalalignment='top', horizontalalignment='left', rotation=90, color=fjOrange)
            else:
                ax2.text(xl, Fe[1]+dF, f"{Fe[1]:2.2f}", fontsize=9, verticalalignment='bottom', horizontalalignment='left', rotation=90, color=fjOrange)
    else:
        raise ValueError("Unknown element type.")

    # plot nodes
    ax2.plot(N, np.zeros(N.size), 'ro')

    # plot node numbers
    for i in range(0,numN):
        ax2.text(N[i], -dF, str(i+1), fontsize=9, verticalalignment='top', horizontalalignment='center')

    ax1.axis('off')
    ax1.set_title('Displacement', fontsize=9)
    ax2.axis('off')
    ax2.set_title('Axial force', fontsize=9)
    
    return fig


if __name__ == "__main__":
    pass
    #N, E, eType, BC, F, p, EA = example_discretization()
    #u = solve_fem_1d(N, E, eType, BC, F, p, EA)
    #fig = post_proc(N, E, u, eType, F, EA)
    #plt.show()
