testprob for meshtal module
1  1  -2.3 (1 -2 3 -4 5 -6)   $ Center Concrete cube
10  2  -7.8 (-10)   $ Middle Iron-Cu ball
11  2  -7.8 (-11)   $ Corner Iron-Cu ball
12  2  -7.8 (-12)   $ Corner Iron-Cu ball
13  2  -7.8 (-13)   $ Corner Iron-Cu ball
14  2  -7.8 (-14)   $ Corner Iron-Cu ball
20  3  -9   (25 -20 -400)  $ Middle Cu cylinder
21  3  -9   (25 -21 -400)  $ Middle Cu cylinder
22  3  -9   (25 -22 -400)  $ Middle Cu cylinder
23  3  -9   (25 -23 -400)  $ Middle Cu cylinder
24  3  -9   (25 -24 -400)  $ Middle Cu cylinder
30  1  -1   -30           $ Concrete cubes  
31  1  -1   -31           $ Concrete cubes  
32  1  -1   -32           $ Concrete cubes  
33  1  -1   -33           $ Concrete cubes  
34  1  -1   -34           $ Concrete cubes  
100 0  (100 -200 300 -400 500 -600 #1)
      (#10 #11 #12 #13 #14 
       #20 #21 #22 #23 #24 $ Outer space
       #30 #31 #32 #33 #34) $ Outer space
99999 0 (-100:200:-300:400:-500:600) $ Graveyard

C Surfaces
1 PX -10
2 PX 10
3 PY -10
4 PY 10
5 PZ -10
6 PZ 10
10 S 20  0  0  5
11 S 20  15  15  5
12 S 20  15 -15  5
13 S 20 -15  15  5
14 S 20 -15 -15  5 
20 C/Y   0  0  5
21 C/Y   15  15  5
22 C/Y   15 -15  5
23 C/Y  -15  15  5
24 C/Y  -15 -15  5
25 PY  21
30 RPP -4    4  -4   4  15  25
31 RPP  20  25  20  25  15  25
32 RPP  20  25 -25 -20  15  25
33 RPP -25 -20  20  25  15  25
34 RPP -25 -20 -25 -20  15  25
100 PX -25
200 PX  25
300 PY -25
400 PY  25
500 PZ -25
600 PZ  25

c materials
m1 8016  0.55 14028  0.18 20040 0.27
m2 26056 0.88 29063 0.12  
m3 29063 0.69 29065 0.31
Mode n P
IMP:n 1 16R 0
SDEF x D1 y D2 Z D3 $ VEC 1 0 0 DIR 1
SI1 -25 25
SP1 0 1
SI2 -25 25
SP2 0 1
SI3 -25 25
SP3 0 1
nps 1e5
fmesh4:N origin -25 -25 -25 geom=XYZ
     IMESH 25
     IINTS 5
     JMESH 25
     JINTS 5
     KMESH 25
     KINTS 5
     EMESH 0.1 1 20
fmesh14:N origin -25 -25 -25 geom=XYZ
     IMESH 25
     IINTS 5
     JMESH 25
     JINTS 5
     KMESH 25
     KINTS 5
FM14 -1
fmesh24:N origin -25 -25 -25 geom=XYZ
     IMESH 25
     IINTS 5
     JMESH 25
     JINTS 5
     KMESH 25
     KINTS 5
FC24 This tally has a comment
FMESH104:N origin -25 0 0 geom=Cyl
     AXS=1 0 0 VEC=0 1 0
     IMESH 25
     IINTS 5
     JMESH 50
     JINTS 5
     KMESH  0.5 1
     KINTS 4 1 
void
PTRAC EVENT=SRC FILE=ASC, MAX = 5e4
