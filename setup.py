from distutils.core import setup

setup (name = 'dolfin_adjoint',
       version = '1.7dev',
       description = 'Automatically derive the discrete adjoint of DOLFIN models',
       author = 'The dolfin_adjoint team',
       author_email = 'patrick.farrell@maths.ox.ac.uk',
       packages = ['fenics_adjoint', 'dolfin_adjoint', 'dolfin_adjoint.optimization',
                   'firedrake_adjoint'],
       package_dir = {'dolfin_adjoint': 'dolfin_adjoint',
                      'fenics_adjoint': 'fenics_adjoint',
                      'firedrake_adjoint': 'firedrake_adjoint'})
