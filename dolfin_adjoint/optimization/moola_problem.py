
__all__ = ["MoolaOptimizationProblem"]

def MoolaOptimizationProblem(rf, memoize=True):
    """Build the moola problem from the OptimizationProblem instance."""

    try:
        import moola
    except ImportError:
        print("You need to install moola. Try `pip install moola`")
        raise

    class Functional(moola.Functional):
        latest_eval_hash = None
        latest_eval_eval = None
        latest_eval_deriv = None

        def __call__(self, x):
            ''' Evaluates the functional for the given control value. '''

            if memoize:
                hashx = hash(x)

                if self.latest_eval_hash != hashx:
                    self.latest_eval_hash = hashx
                    self.latest_eval_eval = rf(x.data)
                    self.latest_eval_deriv = None
                    moola.events.increment("Functional evaluation")
                else:
                    #print  "Using memoised functional evaluation"
                    pass

                return self.latest_eval_eval

            else:
                moola.events.increment("Functional evaluation")
                return rf(x.data)


        def derivative(self, x):
            ''' Evaluates the gradient for the control values. '''

            if memoize:
                # check if the cache is still valid
                self(x)

            if memoize and self.latest_eval_deriv is not None:
                moola.events.increment("Cached derivative evaluation")
                return self.latest_eval_deriv

            moola.events.increment("Derivative evaluation")
            out = [moola.DolfinDualVector(d) for d in rf.derivative(forget=False)]

            if isinstance(x, moola.DolfinPrimalVector):
                deriv = out[0]
            else:
                deriv = moola.DolfinDualVectorSet(out)

            if memoize:
                self.latest_eval_deriv = deriv

            return deriv


        def hessian(self, x):
            ''' Evaluates the gradient for the control values. '''

            self(x)

            def moola_hessian(direction):
                assert isinstance(direction, (moola.DolfinPrimalVector,
                                              moola.DolfinPrimalVectorSet))
                moola.events.increment("Hessian evaluation")
                hes = rf.hessian(direction.data)[0]
                return moola.DolfinDualVector(hes)

            return moola_hessian

    functional = Functional()
    return moola.Problem(functional)
