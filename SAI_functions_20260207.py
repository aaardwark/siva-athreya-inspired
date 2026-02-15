import numpy as np
import matplotlib.collections
import time

rng = np.random.default_rng()


def validate(n,v,e):
    if not (isinstance(v, np.ndarray) and isinstance(e, np.ndarray)):
        raise TypeError('One or both state arrays are not of type ndarray')
    if not (np.bool == v.dtype == e.dtype):
        raise TypeError('One or both of the state arrays are of dtype other than np.bool')
    if not (v.shape == (n,) and e.shape == (n,n)):
        raise ValueError(f'Vertex and edge arrays must have shapes ({n},) and ({n}, {n}) respectively')


def init_state(n_vertices, p_vertexType0, p_edgeConnected):
    """ 
    _ = rng.random((n_vertices,n_vertices)) < p_edgeConnected
    # True w/ chance p
        # __ = np.tril(_, k=-1) 
        # # all 'True's from below main diagonal
        # edges_init = __ | __.T
        # # OR; mirrors to make symmetric 
    edges_init = (tmp := np.tril(_, k=-1) ) | tmp.T
    # take lower triangle and OR it with its transpose
    """
    vertices_init = rng.random(n_vertices) > p_vertexType0
    
    edges_init = np.tril(rng.random((n_vertices,n_vertices)) < p_edgeConnected)
    # independently decides each edge, one copy only of each
    # not symmetric - main diagonal upwards is 0
    edges_init |= edges_init.T
    # OR it with its transpose to make symmetric

    return vertices_init, edges_init


proportion = lambda sub, total: (sub + 1/2)/(total + 1)
# a modified ("tweaked") quantifier of proportion
# avoids divBy0 and 0.0 or 1.0 (deterministic, special/round) values
# Symmetric in that f(U, N) = 1 - f(N-U, N)


def next_state(n_vertices, vertices, edges, type_switching_exponents):
    n_type0, n_type1 = np.count_nonzero((~vertices, vertices), axis=1)
    # counts of type0 and type1 vertices respvly
    edges_bw_unlike =  edges & (vertices ^ vertices[:,np.newaxis])
    # [i,j] True if edge ij is on AND vertices i,j are unlike

    i,j = np.indices(edges.shape, sparse=True)
    # (nV, 1) and (1, nV) arrays respvly, of integers from 0
    # broadcastable to common shape (nV,nV)
    # a[:] and a[:, None] would be the same, where a = arange(nV)


    nNeighbours = np.count_nonzero(edges, axis=1)
    # count of ON edges in each row
    nUnlikeNeighbours = np.count_nonzero(edges_bw_unlike, axis=1)
    # count for each row of ON edges between unlike vertices
    nMutualNeighbours = np.count_nonzero(edges[i,:] & edges[j,:], axis=-1)
    # count of indices that are True in both E[i,:] and E[j,:]
    # gives number of neighbours common to vertices i, j
    # array is (nV,nV,nV) of which I sum along last (3rd) dimension. first two come from i,j


    unlikes_twpr = proportion(nUnlikeNeighbours, nNeighbours)
    # twpr of unlike neighbours against total neighbours for each vertex
    mutual_twpr = proportion(nMutualNeighbours, nNeighbours)
    # twpr of mutual neighbours against total neighbours of first vertex
    # for every (ordered) pair of vertices
    type0_twpr = proportion(n_type0, n_vertices)
    type1_twpr = proportion(n_type1, n_vertices)
    # twprs of each vertex type against total number of vertices
    opptype_twprs = np.where(vertices, type0_twpr, type1_twpr)
    # map each vertex to the twpr of its opposite type (for 'affinity')


    vertex_change_ps = unlikes_twpr ** np.array(type_switching_exponents)[vertices.astype(np.uint8)]
    # go from proportion of unlike influence to probability of change
    # exponent is type-specific; introduces asymmetry, concavity in the function
    vertex_changes = rng.random(n_vertices) < vertex_change_ps
    # actual list of which vertices switch and which don't
    vertices_next = np.where(vertex_changes, ~vertices, vertices)
    # choose from inverted if changes, else from original
    # (changes XOR current) is equivalent, can verify w/ truthtable


    connectivities = (mutual_twpr + mutual_twpr.T)/2
    # taking the mean with its transpose
    # value in-between (Mutual/nNeighboursA) and (Mutual/nNeighboursB) 
    affinities = np.sqrt(opptype_twprs * opptype_twprs[:, np.newaxis])
    # for every edge between i,j vertices, product of opposite twprs of i,j
    # _'s column vector form multiplied by its row vector form
    p_off = 1 - affinities
    # chance of each edge turning off, assuming it's already on
    p_on = np.float_power(connectivities, 1/affinities)
    # chance of each edge turning on, assuming it's off
    p_flip = np.where(edges, p_off, p_on)
    # chance of each edge changing (symmetric array)
    # which is p_off for on edges, and p_on for off edges
    edges_changes = rng.random(edges.shape) < p_flip
    # array of which edges change and which don't
    # asymmetric because of independent random samples
    # but that's fine because we'll ignore the upper triangle totally
    edges_next = np.tril(np.where(edges_changes, ~edges, edges), k=-1)
    # choose from inverted or existing based on flip (again, can use XOR or .where)
    # and discard everything from the main diagonal upwards
    edges_next |= edges_next.T
    # paste the lower triangle onto the upper triangle for symmetry

    return vertices_next, edges_next
 


def save_state_sequence(params={}, **kwparams):
    fname = 'SAIsim-' + time.strftime('%Y%m%d_%H%M', time.localtime())
    np.savez(fname, **params, **kwparams, allow_pickle=False)


def load(timestamp):
    with np.load('SAIsim-' + timestamp + '.npz') as npz:
        sim_data = {}
        for f in npz.files:
            o = npz[f]
            if o.shape == ():
                sim_data[f] = o.item()
            else:
                sim_data[f] = o
    return sim_data




def render_plot(_ax, n_vertices, vertices, edges):
    # arrange vertices clockwise from top
    theta = np.linspace(np.pi*1/2, np.pi*-3/2, n_vertices, endpoint=False)
    x,y = np.cos(theta), np.sin(theta)
    
    """
    LineCollection takes an (nE, nPoints, 2coords) array
    I need: 
    a list of nE edges
        each edge: a list of 2 points
            each point: a list of 2 coordinates
    """
    startV, endV = np.where(np.tril(edges))
    # nE start nodes, nE end nodes
    startcoords = np.stack((x[startV], y[startV]), axis=1)
    # (nE, 2) two columns of (startX, startY)
    endcoords = np.stack((x[endV], y[endV]), axis=1)
    # ditto for endpoints
    edge_coords = np.stack((startcoords, endcoords), axis=1)
    # cross section of 2x2 as seen from 'above' 
    # pushing to axis 1 means the coord dimension gets shifted to axis 2, as required
    
    edges = matplotlib.collections.LineCollection(edge_coords, color='darkgray', linewidth=0.5)
    type0, type1 = np.nonzero(~vertices), np.nonzero(vertices)
    marker_basestyle = {'s': 5**2, 'edgecolors': 'k', 'linewidth': 1, 'zorder': 3}

    _ax.clear()

    # sup_fig = _ax.get_figure(True)
    # dpi, figsize = sup_fig.get_dpi(), sup_fig.get_size_inches()
    # # potentially make marker size and line widths dependent on these


    _ax.add_collection(edges) 
    _ax.scatter(x[type0], y[type0], color='r', **marker_basestyle)
    _ax.scatter(x[type1], y[type1], color='b', **marker_basestyle)
    _ax.set(xticks=(), yticks=())
    return _ax

