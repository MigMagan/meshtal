# -*- coding: utf-8 -*-
# Created on Wed Apr 14 17:20:12 2021
""""Module to sample densities in an MCNP mesh using ptrac.
@author: Miguel Magán
"""
from math import cos, sin, pi, sqrt, log, atan2
import meshtal as mt
import tracer
import cell as cel
import numpy as np
from tqdm import tqdm
from pyne import mcnp
import sparse 


def read_ptrac_head(ptrac_file):
    """Read and return the headers of an open ASCII ptrac file, ptrac_file"""
    if ptrac_file.readline() != "   -1\n":  # Not a ptrac file, or not at the beginning
        print("This is not a ptrac file or is not set at the start, quitting")
        return None
    lin = ptrac_file.readline()  # 2nd line, needs tokenization
    KOD = lin[0:8]  # Using MCNP variable names, hence the capitalization
    VER = lin[8:13]
    if  "6" in VER  or KOD == "mcnpx   ":
        dat_len = 28
    else:
        dat_len = 8
    LODDAT = lin[13:13+dat_len]
    IDTM = lin[13+dat_len:32+dat_len]
    AID = ptrac_file.readline()
    lin = ptrac_file.readline ()  # 3rd line, fun starts
    tokens = [int(float(lin[n:n+12])) for n in range(1, 120, 12)]
    m = tokens[0]
    n = np.zeros((m), dtype=int)
    v = [None]*m
    i = 0
    tokens = iter(tokens[1:])
    while True:
        try:
            n[i] = next(tokens)
            v[i] = np.zeros((n[i]), dtype=int)
            for j in range(n[i]):
                try:
                    v[i][j] = next(tokens)
                except StopIteration:
                    lin = ptrac_file.readline ()  #  read and tokenize next line
                    tokens = [int(float(lin[j:j+12])) for j in range(1, 120, 12)]
                    tokens = iter(tokens)
                    v[i][j] = next(tokens)
            i+=1
            if i == m:
                break
        except StopIteration:
            lin = ptrac_file.readline ()  #  read and tokenize next line
            tokens = [int(float(lin[n:n+12])) for n in range(1, 120, 12)]
            tokens = iter(tokens)
    lin = ptrac_file.readline()
    N1to20 = [int(lin[n:n+5]) for n in range(1, 100, 5)]
    # Read the next files to fill the list of variables L1to11 and we are done
    L1to11 = [None]*11
    Ntot = sum(N1to20[:11])
    Ltokens = np.zeros((Ntot), dtype=int)
    lin = ptrac_file.readline()
    i =0
    j = 1
    while True:
        try:
            Ltokens[i] = int(lin[j:j+4])
            i+=1
            j+=4
            if i == Ntot:
                break
        except ValueError:
            lin = ptrac_file.readline()
            j = 1
    j = 0
    for i in range(11):
        L1to11[i] = Ltokens[j:j+N1to20[i]]
        j = j +N1to20[i]
    return {"KOD": KOD, "VER": VER, "LODDAT": LODDAT, "IDTM": IDTM, "AID": AID , "n":n, "v": v,\
            "N1to20": N1to20, "L1to11": L1to11}


def readnextev(f):
    """Read and return the next event on an open ASCII ptrac file. Return a dictionary with
    SOME of the data. Enhance this function as needed"""
    lin = f.readline()
    tokens = lin.split()
    nextev =  int(tokens[0])
    lin = f.readline()
    ncl = int(tokens[-2])
    tokens = lin.split()
    x, y, z = tokens[0:3]
    return {"next": nextev, "x":x, "y":y, "z":z, "ncl":ncl }


def get_ptrac_src(ptrac_file, pformat="ASCII"):
    """ Return an array of Nx3 with the X, Y, Z coordinates and cell of all src events
    in a ptrac_file. Use pformat="BIN" to read a binary PTRAC """
    if pformat.capitalize() == "Ascii":
        f = open(ptrac_file, 'r')
        head = read_ptrac_head(f)
        npart = head["v"][5][0]
        # print('Max number of particles in ptrac:', npart)  # Notice this is total, SRC may be less
        ptrac_completo=np.zeros((npart, 4))
        for j in tqdm(range(npart), position=0, unit="part", unit_scale=True):
            line = f.readline()
            if not line:
                print ("ptrac apparently stopped at nps=", j)
                break  # end of file
            tokens = line.split()
            Event_ID = int(tokens[1])
            event = readnextev(f)
            if Event_ID == 1000:
                ptrac_completo[j] = [event["x"], event["y"], event["z"], event["ncl"]]
            while event["next"] != 9000:
                event = readnextev(f)
        else:
            j = j +1  # to return the full ptrac and not cut last nps
        return ptrac_completo[:j]
    if pformat.capitalize() == "Bin":
        p = mcnp.PtracReader(ptrac_file)
        npart = 0  # Pyne binary reader does not return the MAX parameter.
        event = {}
        while True:
            try:
                p.read_nps_line()
                if p.next_event == 1000:
                    npart+=1
            except EOFError:
                break  # no more entries
            while p.next_event != 9000:
                p.read_event_line(event)
        print("found {0} events".format(npart))
    # reload and fill data
        ptrac_completo=np.zeros((npart, 4))
        p = mcnp.PtracReader(ptrac_file)
        i = 0
        while True:
            try:
                p.read_nps_line()
            except EOFError:
                break  # no more entries
            while p.next_event != 9000:
                p.read_event_line(event)
                if (event["event_type"] == 1000):
                    ptrac_completo[i] = [event['xxx'], event['yyy'], event['zzz'], event['ncl']]
                    i+=1
        return ptrac_completo[:i]


def ptrac_point_sample(tally, ptrac_file, pformat="ASCII"):
    """ Return the fraction of cells present in each voxel of mesh tally tally,
    using a ptrac_file for tracing and outp_file for cell info. Result is a sparse matrix
    ptracformat: Format of ptrac, ASCII or BIN"""
    if not hasattr(tally,'value'):
        print("Uh, uh, First argument does not seem like a meshtal object. Quitting.")
        return None
    X1i = tally.iints # numero de divisiones en X
    X2i = tally.jints # numero de divisiones en Y
    X3i = tally.kints # numero de divisiones en Z
  # ============================ Transformacion de coordenadas ========================
    ptrac_var = get_ptrac_src(ptrac_file, pformat=pformat)
    if tally.geom=="Cyl":
        ptrac_var[:,0:3] = mt.conv_Cilindricas_multi(tally.axis, tally.vec,
                                                     tally.origin, ptrac_var[:,0:3])
    maxncells = int(max([point[3] for point in ptrac_var]))+1
    npart = len(ptrac_var)
    X1 = tally.ibins
    X2 = tally.jbins
    X3 = tally.kbins
    data = np.zeros((npart), dtype=int)
    index = np.zeros((4, npart), dtype=int)
    print ("processing particles")
    for j, point  in tqdm(enumerate(ptrac_var), position=0, unit="part", unit_scale=True, 
                          total=npart):
        iX1 = np.searchsorted(X1, point[0])-1
        iX2 = np.searchsorted(X2, point[1])-1
        iX3 = np.searchsorted(X3, point[2])-1
        if iX1 in range(X1i) and iX2 in range(X2i) and iX3 in range(X3i):
            index[:, j] = [iX1, iX2, iX3, point[3]]
            data[j] = 1
    s = sparse.COO(index, data, shape=(X1i, X2i, X3i, maxncells))
    Ss = s.sum(axis=3).todense()
    total_voxel = X1i*X2i*X3i
    total_voxel_low = np.count_nonzero(Ss < 10)
    total_voxel_no = total_voxel-np.count_nonzero(Ss)
    ave_hits = sum(data)/total_voxel
    print('Average hits per voxel: ',ave_hits)
    print(total_voxel_low,' of ', total_voxel,' have less than 10 hits per voxel')
    if total_voxel_no!=0:
        print(total_voxel_no,' of ', total_voxel,' have no hits TAKE CARE!!!')
    return s


def ptrac_ray_trace(tally, ptrac_file, pformat="bin", chunksize=10000, cores=None):
    from multiprocessing import Pool
    """ Return the fraction of cells present in each voxel of mesh tally tally,
    using a ptrac_file for ray tracing and outp_file for cell info. Result is a sparse matrix
    ptracformat"""
    if pformat.capitalize() == "Ascii":
        print ("ASCII reading not implemented for ray tracing")
        return None
    initp, finalp, cells = tracer.get_rays(ptrac_file)
    ibins = tally.ibins
    jbins = tally.jbins
    kbins = tally.kbins
    mesh= tracer.Meshtal(ibins, jbins, kbins)
    rays = len(cells)   
    steps = rays // chunksize +2  # We want a minimum of 2 steps even if rays<chunksize
# Divide the lists into Chunks
    n = np.linspace(0, rays-1, steps, dtype=int)
    init_chunk = [initp[j:n[i+1]] for i, j in enumerate(n[:-1])]
    final_chunk = [finalp[j:n[i+1]] for i, j in enumerate(n[:-1])]
    cell_chunk = [cells[j:n[i+1]] for i, j in enumerate(n[:-1])]
    meshes = [mesh for _ in enumerate(n[:-1])]
    starargs = zip(init_chunk, final_chunk, cell_chunk, meshes)
    nchunks = len(meshes)
    print("Tracing chunks of {0} rays".format(chunksize))
    with Pool(cores) as p:
        s0 = p.starmap(tracer.trace_list, tqdm(starargs, total=nchunks), chunksize=1)
        # s0 = p.imap_unordered(do_work, starargs)
    s = sum(s0)
    Ss = s.sum(axis=3).todense()
    total_voxel = s.shape[0]*s.shape[1]*s.shape[2]
    total_voxel_no = total_voxel-np.count_nonzero(Ss)
    if total_voxel_no!=0:
        print(total_voxel_no,' of ', total_voxel,' have no hits TAKE CARE!!!')
    return s


def romesh(tally, ptrac_file, outp_file, method="pointsample", pformat="Ascii", dumpfile=None):
    """Calculate the density in the meshtal tally voxels using ptrac_file and outp_file.
    Returns a density (pseudo) tally. Optionally dump the result in dumpfile.
    Uses by default point sample, pass method=raytrace to use raytracing"""
    if method.capitalize() == "Pointsample":
        pmatrix = ptrac_point_sample(tally, ptrac_file, pformat=pformat)
    elif method.capitalize() == "Raytrace":
        pmatrix = ptrac_ray_trace(tally, ptrac_file, pformat=pformat)
    else:
        print("Unknown method")
        return None
    # Just to check
    if not hasattr(tally,'value'):
        print("Uh, uh, First argument does not seem like a meshtal object. Quitting.")
        return None
    # Lets get a Sorted cellist
    Cellist = cel.ogetall(outp_file)
    MaxNCell = 0
    for cell in Cellist:
        if cell.ncell > MaxNCell:
            MaxNCell = cell.ncell
    SortedCellist=[None]*(MaxNCell+1)
    for cell in Cellist:
        SortedCellist[cell.ncell]=cell
    print ("Sorted Cell list created")
    print ("composing density matrix")
    iints = tally.iints
    jints = tally.jints
    kints = tally.kints
    ro = np.zeros((iints, jints, kints))
    Ss = pmatrix.sum(axis=3)  # Total points or track length of the sampling/tracing
    for i in tqdm(range(iints), position=0):
        for j in range(jints):
            for k in range(kints):
                for l, cell in enumerate(pmatrix[i, j, k].nonzero()[0]):
                    ro[i, j, k] = SortedCellist[cell].density*pmatrix[i, j, k, cell]/Ss[i, j, k] +ro[i, j, k]
    if dumpfile!=None:
        with open(dumpfile,"w") as df:
            for i in range(iints):
                X = (tally.ibins[i]+tally.ibins[i+1])/2
                for j in range(jints):
                    Y = (tally.jbins[j]+tally.jbins[j+1])/2
                    for k in range(kints):
                        Z = (tally.kbins[k]+tally.kbins[k+1])/2
                        df.write("{0} {1} {2} {3:5.4g}\n".format(X, Y, Z, ro[i,j,k]))
    return ro

