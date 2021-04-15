#!/usr/env/ python
" A test for our ptrac ray tracer"
__author__       =  "Miguel Magan"

import numpy as np
import sparse
from pyne import mcnp
from multiprocessing import Pool
import tqdm
from tqdm.contrib import tzip

class Meshtal: # Dummy meshtal class to run
    def __init__(self, ibins, jbins, kbins):
        iints = ibins.size - 1
        jints = jbins.size - 1
        kints = kbins.size - 1
        self.ibins = ibins
        self.jbins = jbins
        self.kbins = kbins
        self.iints = iints
        self.jints = jints
        self.kints = kints
        self.value = np.zeros((iints, kints, kints))

def dummyfunc(i):
    return 2*i

def is_between(n1, n2, n):
    # Check if n is STRICTLY between n1 and n2, regardless of which one is greater.
    if n1 < n < n2:
        return True
    if n2 < n < n1:
        return True
    else:
        return False

def raytracer(p1, p2, mesh, ncell=0):
#   take a ray from array p1 to array p2, and add the contribution to mesh
    line = p2 - p1
    uvw = (line)/np.linalg.norm(line)
    ibins = mesh.ibins
    jbins = mesh.jbins
    kbins = mesh.kbins
    direction = [np.sign(u) for u in uvw]
#   Calculate starting position
    if p1[0] in ibins or p1[1] in jbins or p1[2] in kbins: # Move this 1um
        p1 = p1 + 1e-4*uvw
    i0 = np.searchsorted(ibins, p1[0]) - 1
    j0 = np.searchsorted(jbins, p1[1]) - 1
    k0 = np.searchsorted(kbins, p1[2]) - 1
#   Calculate ending position
    # i1 = np.searchsorted(ibins, p2[0]) - 1
    # j1 = np.searchsorted(jbins, p2[1]) - 1
    # k1 = np.searchsorted(kbins, p2[2]) - 1
#    Calculate the collision planes
    # col_planes1 = []
    col_dists1 = []
    # col_planes2 = []
    col_dists2 = []
    # col_planes3 = []
    col_dists3 = []
    if uvw[0] != 0:
        # col_planes1 = [[x, None, None]  for x in ibins if is_between(p2[0], p1[0], x)]
        col_dists1 = [(x-p1[0])/uvw[0] for x in ibins if is_between(p2[0], p1[0],x)]
    if uvw[1] != 0:
        # col_planes2 = [[None, y, None]  for y in jbins if is_between(p2[1], p1[1], y)]
        # col_dists2 = [(y[1]-p1[1])/uvw[1] for y in col_planes2]
        col_dists2 = [(y-p1[1])/uvw[1] for y in jbins if is_between(p2[1], p1[1],y)]
    if uvw[2] != 0:
        # col_planes3 = [[None, None, z]  for z in kbins if is_between(p2[2], p1[2], z)]
        # col_dists3 = [(z[2]-p1[2])/uvw[2] for z in col_planes3]
        col_dists3 = [(z-p1[2])/uvw[2] for z in kbins if is_between(p2[2], p1[2],z)]
    # cplanes = col_planes1 + col_planes2 + col_planes3
    cdists = np.array(col_dists1 + col_dists2 + col_dists3 + [np.linalg.norm(line)])
    index_dist = np.argsort(cdists)
    
# Build direction matrix. Not very elegant TBH
    cdir = np.zeros((index_dist.size, 3), dtype=int)
    ni = len(col_dists1)
    nj = len(col_dists2)
    nk = len(col_dists3)
    cdir[0:ni] = [direction[0], 0, 0]
    cdir[ni:ni+nj] = [0, direction[1], 0]
    cdir[ni+nj:ni+nj+nk] = [0, 0, direction[2]]
# Build cell matrix
    cells = np.zeros((index_dist.size, 3), dtype=int)
    cells[0] = [i0, j0, k0]
    for i, index in enumerate(index_dist[:-1]):
        cells[i+1] = cells[i] + cdir[index]
    cdists.sort()
    # Trim cdists and cells to take out points outside the mesh
    for i, cell in enumerate(cells):
        if cell[0] in range(mesh.iints):
            if cell[1] in range(mesh.jints):
                if cell[2] in range(mesh.kints):
                    first = i
                    break
    else:
        return None, None #Iterator is exhausted and no point was found in the mesh
    for i, cell in enumerate(cells[::-1]):
        if cell[0] in range(mesh.iints):
            if cell[1] in range(mesh.jints):
                if cell[2] in range(mesh.kints):
                    last = index_dist.size-i
                    break
    # print(cells)
    # print(cdir)
    cdists = np.diff(cdists[first:last], prepend=0)
    cells = cells[first:last]
    # print(cells)
    index = np.zeros((4, cells.shape[0]), dtype=int)
    index[0:3] = np.transpose(cells)
    index[3,:] = ncell
    return index, cdists


def get_rays(ptrac_file):
    """
    Read a binary ptrac_file from mcnp, return three arrays: Initial points, Final points, and
    cell numbers for those events
    """
    # 1st pass to determin number of events
    p = mcnp.PtracReader(ptrac_file)
    nevents = 0
    event = {}
    while True:
        try:
            p.read_nps_line()
        except EOFError:
            break  # no more entries
        p.read_event_line(event)
        while p.next_event != 9000:
            p.read_event_line(event)
            nevents+=1
    print("found {0} events".format(nevents))
    # reload and fill data
    init_points = [None]*nevents
    final_points = [None]*nevents
    cells = [None]*nevents
    p = mcnp.PtracReader(ptrac_file)
    i = 0
    while True:
        try:
            p.read_nps_line()
        except EOFError:
            break  # no more entries
        p.read_event_line(event)
        init_points[i] = np.array([event['xxx'], event['yyy'], event['zzz']])
        c = int(event['ncl'])
        while p.next_event != 9000:
            p.read_event_line(event)
            final_points[i] = np.array([event['xxx'], event['yyy'], event['zzz']])
            cells[i] = c
            # Take these values for next point
            i+=1
            if i == nevents:
                break
            init_points[i] = np.array([event['xxx'], event['yyy'], event['zzz']])
            c = int(event['ncl'])
    
    return init_points, final_points, cells

def trace_list(initp, finalp, cells, mesh):
    """ helper function that takes init points, final points, and cells arrays, and returns a 
    sparse matrix with the tracing in mesh. Notice that the list should be limited in length
    as no chunking is performed"""
    ncells = max(cells)
    index = [[] for _ in range(4)]
    data = []
    for i, cell in enumerate(cells):
        if (initp[i] == finalp[i]).all():
            continue  # Coincident points usually mean termination after entering IMP=0 zone
        a, b = raytracer(initp[i],finalp[i], mesh, cell)
        if a is not None:
            for j in range(4):
                for k in a[j]:
                    index[j].append(k)
            for j in b:
                 data.append(j)
    s = sparse.COO(index, data, shape= (mesh.iints, mesh.jints, mesh.kints, ncells+1))
    return s


def test(n):
    index = [[] for _ in range(4)]
    data = []
    ibins = np.arange(0,101,10)
    jbins = np.arange(0,101,10)
    kbins = np.arange(0,101,10)
    mesh = Meshtal(ibins, jbins, kbins)
    for i in range(n):
        ncell = np.random.randint(1000)
        p1 = np.random.rand(3)*110-5
        p2 = np.random.rand(3)*110-5
        # print ("{0} {1}\n".format(p1, p2))
        a, b = raytracer(p1, p2, mesh, ncell)
        if a is not None:
            for j in range(4):
                for i in a[j]:
                    index[j].append(i)
            for i in b:
                data.append(i)
    return index, data


def test2(ptrac_file, cores=None, chunksize=10000):
    initp, finalp, cells = get_rays(ptrac_file)
    ibins = np.arange(-110,591,10)
    jbins = np.arange(-100,101,10)
    kbins = np.arange(-100,101,10)
    mesh = Meshtal(ibins, jbins, kbins)
    rays = len(cells)
    steps = rays // chunksize
# Divide the lists into Chunks
    n = np.linspace(0, rays-1, steps, dtype=int)
    init_chunk = [initp[j:n[i+1]] for i, j in enumerate(n[:-1])]
    final_chunk = [finalp[j:n[i+1]] for i, j in enumerate(n[:-1])]
    cell_chunk = [cells[j:n[i+1]] for i, j in enumerate(n[:-1])]
    meshes = [mesh for _ in enumerate(n[:-1])]
    starargs = zip(init_chunk, final_chunk, cell_chunk, meshes)
    # pbar = tqdm.tqdm(total=len(init_chunk))
    nchunks = len(meshes)
    print("Tracing chunks of {0} rays".format(chunksize))
    with Pool(cores) as p:
         s0 = p.starmap(trace_list, tqdm.tqdm(starargs, total=nchunks), chunksize=1)
         # s0 = p.imap_unordered(do_work, starargs)
    s = sum(s0)
    return s
