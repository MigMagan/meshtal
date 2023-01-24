#!/usr/env/ python
" A test for our ptrac ray tracer"
__author__       =  "Miguel Magan"

from multiprocessing import Pool
import tqdm
from tqdm.contrib import tzip
from math import atan2, pi
from pyne import mcnp
import numpy as np
import sparse

class Meshtal:
    "This is just a Dummy meshtal class with the mesh"
    def __init__(self, ibins, jbins, kbins, geom="XYZ"):
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
        self.geom = geom
        if self.geom == "Cyl":
            self.axis = np.array([0, 0, 1])  # Axis
            self.vec = np.array([1, 0, 0])   # Azimuthal vector
            self.origin = np.array([0, 0, 0])

def is_between(n1, n2, n):
    # Check if n is STRICTLY between n1 and n2, regardless of which one is greater.
    return (n1 < n < n2) or (n2 < n < n1)

def __cyl_raytracer(p1, p2, mesh, ncell=0):
    """Take a ray from numpy array p1 to numpy array p2 and add the contribution to a cylindrical
    mesh. The vector the points define MUST be parallel to the axis, and the mesh must have
    normalized axis and vec"""
    line = p2 -p1
    if (np.cross(line, mesh.axis)/(np.linalg.norm(line)) > 1E-5).any():
        print ("For ray tracing in cylindrical mesh, rays must be parallel to axis!")
        return None, None
#   Calculate non-changing Ro and Theta
    p0 = p1 - mesh.origin
    yvec = np.cross(mesh.axis, mesh.vec)
    rvec = p0 - mesh.axis*np.dot(p0, mesh.axis)
    r = np.linalg.norm(rvec)
    if r> mesh.ibins[-1] or r<mesh.ibins[0]:
        print("ray traced outside of mesh")
        return None, None
    t = atan2(np.dot(rvec, yvec), np.dot(rvec,mesh.vec)) % (2*pi)
    t = t/ (2*pi) # To have it in revolutions
    rindex = np.searchsorted(mesh.ibins, r)-1
    tindex = np.searchsorted(mesh.kbins, t, side = "right")-1
    h1 = np.dot(p0, mesh.axis)  # initial height position
    h2 = np.dot(p2-mesh.origin, mesh.axis)  # Final height position
    h1 = max(mesh.jbins[0], min(h1, mesh.jbins[-1]))  # limit h1 and h2 to the mesh
    h2 = max(mesh.jbins[0], min(h2, mesh.jbins[-1]))
    if h1>h2:  # turn around if vector is counter-parallel to axis
        h1, h2 = h2, h1
    elif h1 == h2:
        return None, None
    hindex1 = np.searchsorted(mesh.jbins, h1, side="right")-1
    hindex2 = np.searchsorted(mesh.jbins, h2)-1
    index = np.zeros((4, hindex2-hindex1+1), dtype="int32")
    index[0,:] = rindex
    index[1,:] = list(range(hindex1, hindex2+1))
    index[2,:] = tindex
    index[3,:] = ncell
    cdists = np.diff(mesh.jbins[hindex1+1:hindex2+1], prepend=h1, append=h2)
    return index, cdists

def __xyz_raytracer(p1, p2, mesh, ncell=0):
    "take a ray from array p1 to array p2, and add the contribution to a cartesian mesh"
    if (p1==p2).all():
        return None, None
    line = p2 - p1
    uvw = (line)/np.linalg.norm(line)
    # ints = [mesh.iints, mesh.jints, mesh.kints]
    bins = [mesh.ibins, mesh.jbins, mesh.kbins]
    sides = ["left" if u<0 else "right" for u in uvw]
    direction = [int(i) for i in np.sign(uvw)]
    ijk0 = [np.searchsorted(bins[i], p1[i], side=sides[i])- 1 for i in range(3)]
    col_dists = [[] for _ in range(3)]
    for i in range(3):
        if uvw[i] !=0:
            col_dists[i] = np.array([(x-p1[i]) for x in bins[i] if is_between(p2[i], p1[i], i)]) 
            col_dists[i]/=uvw[i]
    # cplanes = col_planes1 + col_planes2 + col_planes3
    cdists = np.concatenate((col_dists[0], col_dists[1], col_dists[2], [np.linalg.norm(line)]))
    # print(cdists)
    index_dist = np.argsort(cdists)
# # Build direction matrix. Not very elegant TBH
    cdir = np.zeros((index_dist.size, 3), dtype="int8")
    ni = len(col_dists[0])
    nj = len(col_dists[1])
    nk = len(col_dists[2])
    cdir[0:ni] = [direction[0], 0, 0]
    cdir[ni:ni+nj] = [0, direction[1], 0]
    cdir[ni+nj:ni+nj+nk] = [0, 0, direction[2]]
# # Build cell matrix
    cells = np.zeros((index_dist.size, 3))
    cells[0] = ijk0
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
    cdists = np.diff(cdists, prepend=0)[first:last]
    cells = cells[first:last]
    index = np.zeros((4, cells.shape[0]), dtype="int32")
    index[0:3] = np.transpose(cells)
    index[3,:] = ncell
    return index, cdists

def raytracer(p1, p2, mesh, ncell=0):
    if not isinstance(p1, np.ndarray) or not isinstance(p2, np.ndarray):
        raise TypeError("only numpy arrays con be used as points")
    if any([np.size(p1) !=3, np.size(p2) !=3]):
        print("trying to trace points {0} and {1}".format(p1, p2))
        raise IndexError("points don't have 3 dimensions")
    if mesh.geom=="XYZ":
        return __xyz_raytracer(p1, p2, mesh, ncell)
    if mesh.geom=="Cyl":
        return __cyl_raytracer(p1, p2, mesh, ncell)
    print("unknown geometry type")
    return None

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
        nevents+=1
        while p.next_event != 9000:
            p.read_event_line(event)
            nevents+=1
    print("found {0} events".format(nevents))
    # reload and fill data
    init_points = [None]*nevents
    final_points = [None]*nevents
    cells = [None]*nevents
    p = mcnp.PtracReader(ptrac_file)
    p.read_nps_line()
    for i in tqdm.tqdm(range(nevents), position=0, unit="event", unit_scale=True):
        p.read_event_line(event)
        init_points[i] = np.array([event['xxx'], event['yyy'], event['zzz']])
        cells[i] = int(event['ncl'])
        if event["event_type"] != 1000:
            final_points[i-1] = np.array([event['xxx'], event['yyy'], event['zzz']])
        else:
            final_points[i-1] = np.array([None, None, None])
        if p.next_event == 9000:
            try:
                p.read_nps_line()
            except EOFError:
                break  # no more entries
    return init_points, final_points, cells


def postprocess_rays(init_points, final_points, cells, inf_distance=1000):
    """Post-process the points given to either assign coordinates to the null points
    or delete them if they can not be tracked"""
    r0 = []
    r1 = []
    r2 = []
    print("postprocessing ray list")
    for i, point in tqdm.tqdm(enumerate(final_points), unit ="rays", position=0, unit_scale=True):
        if not point.all():
            if (final_points[i-1] == init_points[i]).all():  # tracable ray
                direction = final_points[i-1] - init_points[i-1]
                r0.append(init_points[i])
                r1.append(init_points[i]+direction*inf_distance/np.linalg.norm(direction))
                r2.append(cells[i])
            else:  # Non-tracable. Discard the point
                pass
        else:
            r0.append(init_points[i])
            r1.append(final_points[i])
            r2.append(cells[i])
    discarded = len(cells) - len(r2)
    print("\n{0} points out of {1} were discarded".format(discarded, len(cells)))
    return r0, r1, r2


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
    initp, finalp, cells = postprocess_rays(initp, finalp, cells)
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
