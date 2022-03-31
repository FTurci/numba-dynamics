import numpy
from numba import double, njit
from atooms.core.utils import Timer

#all in reduced units
rc2 = 2.5**2
rc2i=1.0/rc2
rc6i=rc2i*rc2i*rc2i
ecut=rc6i*(rc6i-1.0)

@njit( (double[:,:], double[:]))
def lennardjones(U, box):

    # Can't use (ndim, npart) = numpy.shape(U)
    # with Numba. No unpacking of tuples.

    ndim = len(U)
    npart = len(U[0])

    F = numpy.zeros((ndim,npart))
    Epot = 0.0
    Vir = 0.0

    for i in range(npart):
        for j in range(i+1,npart):
            X  = U[0, j] - U[0, i]
            Y  = U[1, j] - U[1, i]
            Z  = U[2, j] - U[2, i]

            # Periodic boundary condition
            X  -= box[0] * numpy.rint(X/box[0])
            Y  -= box[1] * numpy.rint(Y/box[1])
            Z  -= box[2] * numpy.rint(Z/box[2])

            # Distance squared
            r2 = X*X + Y*Y + Z*Z
            if(r2 < rc2):
                r2i = 1.0 / r2
                r6i = r2i*r2i*r2i
                Epot = Epot + r6i*(r6i-1.0) - ecut

                ftmp = 48. * r6i*(r6i- 0.5) * r2i

                F[0, i] -= ftmp * X
                F[1, i] -= ftmp * Y
                F[2, i] -= ftmp * Z
                F[0, j] += ftmp * X
                F[1, j] += ftmp * Y
                F[2, j] += ftmp * Z
                Vir += ftmp
    Epot = Epot * 4.0

    return Epot, F, Vir

@njit( (double[:,:], double[:]) )
def pbc(x,box):
    ndim = len(x)
    npart = len(x[0])
    bhalf = box*0.5
    for k in range(ndim):
        for p in range(npart):
            if x[k,p]>bhalf[k]:
                x[k,p]-=box[k]
            elif x[k,p]<-bhalf[k]:
                x[k,p]+=box[k]

class LJVerlet(object):
    def __init__(self,system,timestep):
        self.system = system
        self.timestep = timestep

        self.timer = {
            'evolve': Timer(),
        }


    def run(self,steps):
        # work in reduced units
        _box = self.system.dump('cell.side', dtype='float64', view=True)
        _pos = self.system.dump('particle.position', dtype='float64', view=True, order='F')
        _vel = self.system.dump('particle.velocity', dtype='float64', view=True, order='F')
        dt = self.timestep


        self.timer['evolve'].start()

        (epot, F, Vir) = lennardjones(_pos, _box)

        for _ in range(steps):
            _pos += _vel* dt + 0.5 * F * dt * dt
            F0 = F[:]
            epot, F, Vir = lennardjones(_pos, _box)
            _vel += 0.5 * (F + F0) * dt

            pbc(_pos,_box)

        self.timer['evolve'].stop()

from atooms.system import System
from atooms.simulation import Simulation
from atooms.trajectory import TrajectoryEXYZ
import tqdm
system = System(N=800)
system.density = 0.1
backend = LJVerlet(system,0.002)
sim = Simulation(backend, steps=200)

for p in system.particle:
    p.maxwellian(T=0.8)

with TrajectoryEXYZ("output.exyz","w",) as tj:
    for k in tqdm.tqdm(range(10)):
        sim.run()
        tj.write(sim.system)
