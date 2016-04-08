import libadjoint

class CoeffStore(object):
    '''This object manages the mapping from Dolfin coefficients to libadjoint Variables.
    In the process, it also manages the incrementing of the timestep associated with each
    variable, so that the user does not have to manually manage the time information.'''
    def __init__(self):
        self.coeffs = {}
        self.libadjoint_timestep = 0
        self.str_to_coeff = {}

    def next(self, coeff):
        '''Increment the timestep corresponding to the provided Dolfin
        coefficient and then return the corresponding libadjoint variable.'''

        if not isinstance(coeff, str):
            self.str_to_coeff[str(coeff)] = coeff

        coeff = str(coeff)

        try:
            (timestep, iteration) = self.coeffs[coeff]
            if timestep == self.libadjoint_timestep:
                self.coeffs[coeff] = (timestep, iteration + 1)
            else:
                self.coeffs[coeff] = (self.libadjoint_timestep, 0)
        except KeyError:
            self.coeffs[coeff] = (self.libadjoint_timestep, 0)

        (timestep, iteration) = self.coeffs[coeff]
        return libadjoint.Variable(coeff, timestep, iteration)

    def __getitem__(self, coeff):
        '''Return the libadjoint variable corresponding to coeff.'''

        if not isinstance(coeff, str):
            self.str_to_coeff[str(coeff)] = coeff

        coeff = str(coeff)

        if not self.coeffs.has_key(coeff):
            self.coeffs[coeff] = (self.libadjoint_timestep, 0)

        (timestep, iteration) = self.coeffs[coeff]
        return libadjoint.Variable(coeff, timestep, iteration)

    def keys(self):
        for i in self.coeffs:
            yield self.str_to_coeff[i]

    def increment_timestep(self):
        self.libadjoint_timestep += 1

    def forget(self, coeff):
        del self.coeffs[str(coeff)]
