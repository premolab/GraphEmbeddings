
import pickle
import networkx as nx
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from transliterate import translit
from datetime import datetime
import subprocess
import sys
import os
import shutil

col = ['b', 'r', 'g', 'm', 'c', 'y', '#56A0D3', '#ED9121', '#00563F', '#062A78', '#703642', '#C95A49',
       '#92A1CF', '#ACE1AF', '#007BA7', '#2F847C', '#B2FFFF', '#4997D0',
       '#DE3163', '#EC3B83', '#007BA7', '#2A52BE', '#6D9BC3', '#007AA5',
       '#E03C31', '#AAAAAA']

def toBigClamFormat(G, file_name):
    if isinstance(G, nx.Graph):
        A = np.array(nx.to_numpy_matrix(G))
    else:
        A = G
    G = nx.Graph(A)
    with file(file_name, 'w') as f:
        [f.write('{}\t{}\n'.format(e[0]+1, e[1]+1)) for e in G.edges()]

def fromBigClamFormat(file_name):
    data = []
    N = 0
    with file(file_name, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            a, b = line.split()
            a, b = int(a), int(b)
            data.append((a, b))
            N = max(N, a, b)
    A = np.zeros(shape=(N+1, N+1))
    for a, b in data:
        A[a][b] = 1
    return nx.Graph(A)


def fromBigClamFormat_sparse(file_name):
    data = []
    with open(file_name, 'r') as f:
        data = [list(map(int, line.split())) for line in f if line[0] != '#']
    G = nx.Graph(data=data)
    return G


def test_example():
    return np.array([[0, 1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 1, 0, 0, 0],
                     [1, 1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 1, 1, 1],
                     [0, 0, 0, 1, 0, 1, 1],
                     [0, 0, 0, 1, 1, 0, 1],
                     [0, 0, 0, 1, 1, 1, 0]])


def draw_groups(A, F, ids=None, names=None, figname = 'NoName', png=True, pdf=False, display=False, svg=False, dpi=2300):
    N, K = F.shape

    C = F > np.sum(A) / (A.shape[0] * (A.shape[0] - 1))
    indx = np.argmax(F, axis=1)
    for i in range(N):
        C[i, indx[i]] = True
    print(F)
    print(C)

    comm = [[] for i in range(N)]
    for x, y in zip(*np.where(C)):
        comm[x].append(y)
    u_comm = np.unique(comm)

    comm2id = []
    for u in u_comm:
        comm2id.append([i for i, c in enumerate(comm) if c == u])

    G = nx.Graph(A)
    plt.figure(num=None, figsize=(10, 10))

    pos = []
    centers = [np.array([0, 1])]
    angle = np.pi / K
    turn = np.array([[np.cos(2*angle), np.sin(2*angle)], [-np.sin(2*angle), np.cos(2*angle)]])
    radius = np.sin(angle)
    new_pos = {i: [] for i in range(N)}

    U, s, V = np.linalg.svd(F.T.dot(F))
    posSVD =[x[0] for x in sorted([x for x in enumerate(U[0])], key= lambda x: x[1])]

    for i in range(K):
        if i + 1 != K:
            centers.append(turn.dot(centers[-1]))

    for i in range(K):
        for key, value in nx.spring_layout(G.subgraph(np.where(C[:, posSVD[i]])[0])).items():# positions for all nodes
            new_pos[key].append(value * radius + 0.8 * centers[posSVD[i]])

    for key in new_pos:
        new_pos[key] = np.sum(np.array(new_pos[key]), axis=0) / (1.5 * len(new_pos[key])) ** 1.2

    for val in comm2id:
        if len(comm[val[0]]) < 2:
            continue
        m = np.mean(np.array([new_pos[x] for x in val]), axis=0)
        for x in val:
            new_pos[x] = 0.8 * len(comm[val[0]]) * (new_pos[x] - m) + m

    nx.draw_networkx_edges(G, new_pos, width=0.25, alpha=0.07)
    nx.draw_networkx_nodes(G, new_pos, node_color='#BBBBBB', node_size=15, alpha=1, linewidths=0)
    for j in range(C.shape[0]):
        k = 0
        for i in range(C.shape[1]):
            if(C[j][i]):
                nx.draw_networkx_nodes(G, new_pos, nodelist=[j], node_color=col[i], node_size=10-1*k,
                                       alpha=0.6, linewidths=0)
                k += 1
    if ids is not None and names is not None:
        labels = {i: ' '.join([str(n) for n in np.where(c)[0]]) + '\n> {} <'.format(translit(names[ids[i]].replace('\u0456', '~'), 'ru', reversed=True)) for i, c in enumerate(C)}
        nx.draw_networkx_labels(G, new_pos, labels, font_size=0.1)
    plt.axis('off')

    if pdf:
        plt.savefig("../plots/{}.pdf".format(figname))
    if png:
        plt.savefig("../plots/{}.png".format(figname), dpi=dpi)
    if svg:
        plt.savefig("../plots/{}.svg".format(figname))
    if display:
        plt.show() # display

def progress(list, update_interval=1):

    """
    display progress for loop list
    :param list: list
    :param update_interval: minimal update iterval for progress
    :return: generator with progress output to stdout
    """
    N = len(list)
    start = datetime.now()
    last = start
    sys.stdout.write("\rProgress: {:.2f}% | ETA/Total: {:.2f}/{:.2f} sec {}".format(0, float('nan'), float('nan'), ' '))
    for index, val in enumerate(list):
        yield val
        time = datetime.now()
        if (time - last).seconds > update_interval:
            sys.stdout.write("\rProgress: {:.2f}% | ETA/Total: {:.2f}/{:.2f} sec {}"
                             .format(100.0 * (index + 1) / N, (time - start).seconds / (index + 1) * (N - 1 - index),
                                     (time - start).seconds / (index + 1) * N, " " * 30))
            sys.stdout.flush()
            last = time


if __name__ == '__main__':
    D = pickle.load(open('../data/vk/3771369.ego'))
    G = nx.Graph(D)
    toBigClamFormat(G, '../data/vk/3771369.bigClam')
    G = fromBigClamFormat('../data/test.bigclam')
    A = np.array(nx.to_numpy_matrix(G))
    pass

def conductance(G, S, T=None, weight=None):
    """Returns the conductance of two sets of nodes.
    Fixed by Zurk

    The *conductance* is the quotient of the cut size and the smaller of
    the volumes of the two sets. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : sequence
        A sequence of nodes in `G`.

    T : sequence
        A sequence of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The conductance between the two sets `S` and `T`.

    See also
    --------
    cut_size
    edge_expansion
    normalized_cut_size
    volume

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    if T is None:
        T = set(G) - set(S)
    num_cut_edges = nx.algorithms.cuts.cut_size(G, S, T, weight=weight)
    volume_S = nx.algorithms.cuts.volume(G, S, weight=weight)
    volume_T = nx.algorithms.cuts.volume(G, T, weight=weight)

    if volume_S == 0:
        return 0
    if volume_T == 0:
        return 1

    return num_cut_edges / min(volume_S, volume_T)

def getSeedCenters(A, K=None, w=10, pool=None):
    G = nx.Graph(A)
    if K is None:
        K = nx.number_of_nodes(G)
    cond = GetNeighborhoodConductance(G, pool=pool)
    local_max = conductanceLocalMin(G, None, cond, pool)
    res = [local_max.pop(0)]

    while len(res) < K and len(local_max) > 0:
        all_nodes = np.any(A[res] != 0, axis=0)
        interseption = A[local_max].dot(all_nodes.T) / np.sum(A[local_max], axis=1)
        k = np.argmin(cond[local_max] + w * interseption)
        res.append(local_max.pop(k))
    return res

def conductanceLocalMin(G, K=None, cond=None, pool=None):
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G)
    if K is None:
        K = len(G)
    InvalidNIDS = []
    if cond is None:
        cond = GetNeighborhoodConductance(G, pool=pool)
    cond = sorted(enumerate(cond), key=lambda x: x[1])
    indx = []
    CurCID = 0
    for ui in range(len(cond)):
        UID = cond[ui][0]
        if UID in InvalidNIDS:
            continue
        indx.append(UID)
        NI = list(G[UID].keys())  # neighbours of UID
        #NI = np.where(NI)[0]
        InvalidNIDS.extend(NI)
        InvalidNIDS.append(UID)
        CurCID += 1
        if CurCID >= K:
            break
    return indx

def getEgoGraphNodes(G, u):
    return [u] + [x for x in G.neighbors(u)]


def GetNeighborhoodConductance_worker(args):
    u, G = args[0], args[1]
    minDeg = 10
    return 1 if G.degree(u, weight='weight') < minDeg else conductance(G, getEgoGraphNodes(G, u))

def GetNeighborhoodConductance(G, minDeg = 10, pool=None):
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G)
    N = len(G)
    Edges2 = 2*len(G.edge)
    if pool is not None:
        NIdPhiV = np.array(pool.map(GetNeighborhoodConductance_worker, ((u, G) for u in G)))
    else:
        NIdPhiV = np.zeros(shape=(N,))
        for u in G:
            NIdPhiV[u] = 1 if G.degree(u, weight='weight') < minDeg else conductance(G, getEgoGraphNodes(G, u))
    return NIdPhiV

def LancichinettiBenchmark(**kwargs):
    """
    Benchmark graphs for testing community detection algorithms from
    Andrea Lancichinetti, Santo Fortunato and Filippo Radicchi1
    http://arxiv.org/pdf/0805.4770.pdf

    This function is wrapper for exe file.
    In future it should be rewritten by Python-C API

    [FLAG]		    [P]
    :param N		number of nodes
    :param k		average degree
    :param maxk		maximum degree
    :param mut		mixing parameter for the topology
    :param muw		mixing parameter for the weights
    :param beta		exponent for the weight distribution
    :param t1		minus exponent for the degree sequence
    :param t2		minus exponent for the community size distribution
    :param minc		minimum for the community sizes
    :param maxc		maximum for the community sizes
    :param on		number of overlapping nodes
    :param om		number of memberships of the overlapping nodes
    :param C        [average clustering coefficient]
    """

    default = { 'N': 1000,
                 'mut': 0.1,
                 'maxk': 50,
                 'k': 30,
                 'om': 2,
                 'muw': 0.1,
                 'beta': 2,
                 't1': 2,
                 't2': 2,
                 'on': 100
                 }

    default.update(kwargs)

    cwd = '../external/Lancichinetti_benchmark/benchmark/'
    #os.chdir('../external/Lancichinetti_benchmark/benchmark')

    with open(cwd+"parameters.dat", 'w') as f:
        f.write('\n'.join('-{} {}'.format(key, default[key]) for key in default))
    with open(cwd+'outputlog', 'wb') as outputlog:
        p = subprocess.Popen('./benchmark -f parameters.dat'.split(), stdout=outputlog, stderr=outputlog, cwd=cwd)
        p.wait()
    with open(cwd+"network.dat", 'rb') as nw:
        G = nx.read_weighted_edgelist(nw)
    with open(cwd+"community.dat") as nw:
         comm_inv = {line.split()[0]: line.split()[1:] for line in nw}
    comm = {}
    for key in comm_inv:
        for x in comm_inv[key]:
            t = int(x)
            if t in comm:
                comm[t].append(key)
            else:
                comm[t] = [key]

    #os.chdir(cwd)

    return G, comm

def draw_loglikelihood_slice(bigClam, F, u, direction, step):
    res = []
    Fres = []
    du = F[u].copy()
    points = np.linspace(-0.5 * step, 1.5 * step, 1501)
    zero_indx = np.where(abs(points) < step * 1e-4)[0][0]
    step_indx = np.where(abs(points - step) < step * 1e-4)[0][0]
    for t in points:
        F[u] = du + t * direction
        F[u][F[u] < 0] = 0
        F[u][F[u] > 1000] = 1000
        res.append(bigClam.loglikelihood(F))
        Fres.append(F.copy())
    ax1 = plt.subplot(311)
    plt.plot(points, res)
    plt.scatter([0.0, step], [res[zero_indx], res[step_indx]], c='r')
    plt.setp(ax1.get_xticklabels(), fontsize=6)
    ax2 = plt.subplot(312, sharex=ax1)
    Fmax = [np.max(f) for f in Fres]
    plt.plot(points, Fmax)
    plt.scatter([0.0, step], [Fmax[zero_indx], Fmax[step_indx]], c='r')
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3 = plt.subplot(313, sharex=ax1)
    Fzeros = [np.sum(f == 0) for f in Fres]
    plt.plot(points, Fzeros)
    plt.scatter([0.0, step], [Fzeros[zero_indx], Fzeros[step_indx]], c='r')
    plt.show()
    return res

def gradient_check(bigClam, F, Fu, u=None):
    D = F.copy()
    D[u] = Fu
    t = bigClam.sumF.copy()
    bigClam.update_sumF(D[u], F[u])
    ans = bigClam.gradient(D, u)
    bigClam.sumF = t
    return ans

def MixedModularity(F, A):
    if isinstance(F, list):
        F_new = np.zeros((A.shape[0], len(F)))
        for indx, comm in enumerate(F):
            for c in comm:
                F_new[c-1, indx] = 1
        F = F_new
    F = F.copy()
    #print F.shape
    #F = F > 0.01 * np.sum(A) / (A.shape[0]*(A.shape[0]-1) )
    Fnorm = np.sum(F, axis=1)
    F[Fnorm!=0,:] = F[Fnorm!=0, :] * (1.0 / Fnorm[Fnorm!=0, None])
    m2 = 1.0 * np.sum(A)
    res = 0.0
    for i in range(F.shape[1]):
        indx = F[:, i] != 0
        degs = np.sum(A[indx], axis=1)[:, None]
        M = A[indx, :][:, indx] - degs.dot(degs.T) / m2
        M = M * F[indx, [i]][:,None].dot(F[indx, [i]][:,None].T)
        res += np.sum(M)
    return res / m2

def GetComms(F, A):
    C = F > (np.sum(A) / (np.mean(A[A != 0]) * A.shape[0] * (A.shape[0] - 1)))
    return [np.nonzero(S)[0]+1 for S in C.T if len(np.nonzero(S)[0]) not in {0, C.shape[1]}]

def Conductance(comms, A):
    G = nx.Graph(A)
    res = [conductance(G, c) for c in comms]
    return res

def MeanConductance(F, A):
    return np.mean(Conductance(F, A))

def MaxConductance(F, A):
    return np.max(Conductance(F, A))

def sorted_t(list):
    return sorted(list, key=lambda x: int(x))

def NMI3(comms, A, Comm_True):
    with open('../external/Lancichinetti benchmark/clu1-3', 'w') as f:
        for comm in comms:
            f.write(" ".join(str(c) for c in sorted_t(comm)))
            f.write('\n')

    with open('../external/Lancichinetti benchmark/clu2-3', 'w') as f:
        for key in Comm_True:
            f.write(" ".join(str(c) for c in sorted_t(Comm_True[key])))
            f.write('\n')
    cwd = os.getcwd()
    os.chdir('../external/Lancichinetti benchmark/')
    with open('outputlog-nmi3', 'wb') as outputlog:
        p = subprocess.Popen('mutual.exe clu1-3 clu2-3', stdout=outputlog, stderr=outputlog)
        p.wait()
    os.chdir(cwd)
    with open('../external/Lancichinetti benchmark/outputlog-nmi3', 'r') as f:
        for line in f:
            res = line.split()
            return float(res[-1])

def NMI(comms, A, Comm_True):

    #min_indx = min(c for comm in comms for c in comm) - 1

    with open('../external/Lancichinetti benchmark/clu1', 'w') as f:
        for indx, comm in enumerate(comms):
            for c in sorted_t(comm):
                f.write("1 {} {}\n".format(c, indx))

    with open('../external/Lancichinetti benchmark/clu2', 'w') as f:
        for key in Comm_True:
            for c in sorted_t(Comm_True[key]):
                f.write("1 {} {}\n".format(c, key))
    cwd = os.getcwd()
    os.chdir('../external/Lancichinetti benchmark/')
    with open('outputlog-nmi', 'wb') as outputlog:
        p = subprocess.Popen('nmi.exe clu1 clu2', stdout=outputlog, stderr=outputlog)
        p.wait()
    os.chdir(cwd)
    with open('../external/Lancichinetti benchmark/outputlog-nmi', 'r') as f:
        for line in f:
            res = line.split()
            if res[0] == 'Multiplex':
                return float(res[-1])

def bigclam_orig(A,K):
    np.fill_diagonal(A, 0)
    cur_path = os.getcwd()
    data_dir = '../external/BigClam/'
    data_file_name = 'test'
    bigClam_data_file_path = os.path.join(data_dir, data_file_name) + '.bigClam'
    bigClam_res_prefix = data_dir
    exe_path = '../external/BigClam/bigclam.exe'

    toBigClamFormat(A, bigClam_data_file_path)

    args = '-i:\"{}\" -o:\"{}\" -c:{} -nt:4'.format(os.path.join(cur_path, bigClam_data_file_path),
                                               os.path.join(cur_path, bigClam_res_prefix),  K)

    with open('../external/BigClam/bigClam_output.log', 'w') as output_f:
        subprocess.call('\"{}\" {}'.format(exe_path, args), stdout=output_f, stderr=output_f)

    with open('../external/BigClam/cmtyvv.txt', 'r') as f:
        res =[[int(x) for x in line.split()] for indx, line in enumerate(f)]
    return res

def toCorpraFormat(A, file_name):
    G = nx.Graph(A)
    with file(file_name, 'w') as f:
        [f.write('{} {} {}\n'.format(e[0]+1, e[1]+1, e[2]['weight'])) for e in G.edges(data=True)]

def copra(A, K=None):
    data_dir = '../external/COPRA/'
    data_file_name = 'test'
    COPRA_data_file_path = os.path.join(data_dir, data_file_name) + '.COPRA'
    java_path = '../external/COPRA/copra.jar'

    toCorpraFormat(A, COPRA_data_file_path)

    args = '-w -v 2 -prop 500000'

    with open('../external/COPRA/COPRA_output.log', 'w') as output_f:
        subprocess.call('java -cp \"{}\" COPRA \"{}\" {}'.format(java_path, COPRA_data_file_path, args),
                        stdout=output_f, stderr=output_f)

    with open('clusters-test.COPRA', 'r') as f:
        res = [[int(x) for x in line.split()] for indx, line in enumerate(f)]
    return res

def CFinder(A, K):
    cur_path = os.getcwd()
    data_dir = '../external/CFinder-2.0.6--1448'
    data_file_name = 'test'
    cFinder_data_file_path = os.path.join(data_dir, data_file_name) + '.cfinder'
    cFinder_licence_file_path = os.path.join(data_dir, 'licence.txt')
    cFinder_res_prefix = os.path.join(data_dir, 'res')
    shutil.rmtree(cFinder_res_prefix)
    exe_path = '../external/CFinder-2.0.6--1448/CFinder_commandline.exe'

    toCorpraFormat(A, cFinder_data_file_path)

    args = '-i \"{}\" -o \"{}\" -l \"{}\" -k 4'.format(os.path.join(cur_path, cFinder_data_file_path),
                                                  os.path.join(cur_path, cFinder_res_prefix),
                                                  os.path.join(cur_path, cFinder_licence_file_path))

    with open('../external/CFinder-2.0.6--1448/CFinder_output.log', 'w') as output_f:
        subprocess.call('\"{}\" {}'.format(exe_path, args), stdout=output_f, stderr=output_f)

    res = []
    with open('../external/CFinder-2.0.6--1448/res/k=4/communities', 'r') as f:
        for indx, line in enumerate(f):
            if line[0] != '#' and line[0] != '\n':
                res.append([int(x) for x in line.split(':')[1].split()])
    return res

def walktrap(A, K):
    import igraph
    G = igraph.Graph.Adjacency((A > 0).tolist())
    G.es['weight'] = A[A.nonzero()]
    clust = G.community_walktrap(weights='weight').as_clustering()
    return [[xt+1 for xt in x] for x in clust]

if __name__ == '__main__':
    A = 2.0 * test_example()
    A[2, 1] = 3.0

    print(walktrap(A, 2))
    print(CFinder(A, 2))
    print(copra(A, 2))
    print(bigclam_orig(A, 2))

    #F = np.array([[0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0]]).T
    #print MeanConductance(GetComms(F, A), A)
    #print NMI3(GetComms(F, A), A, {0: [4, 1, 2, 3], 1: [7, 4, 5, 6]})

    #print bigclam_orig(nx.Graph(A), 2)
    #print copra(A, 2)
    #LancichinettiBenchmark()

