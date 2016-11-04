import backend
from . import utils
from . import adjglobals
from . import solving
import libadjoint
from . import adjlinalg

if hasattr(backend, 'FunctionAssigner'):
    class FunctionAssigner(backend.FunctionAssigner):

        def __init__(self, *args, **kwargs):
            super(FunctionAssigner, self).__init__(*args, **kwargs)

            # The adjoint function assigner with swapped FunctionSpace arguments
            self.adj_function_assigner = backend.FunctionAssigner(args[1], args[0])

        def assign(self, receiving, giving, annotate=None):

            out = backend.FunctionAssigner.assign(self, receiving, giving)
            to_annotate = utils.to_annotate(annotate)

            if to_annotate:
            # Receiving is always a single Function, or a single Function.sub(foo).sub(bar)

            # If the user passes in v.sub(0) to this function, we need to get a handle on v;
            # fetch that now
                receiving_super = get_super_function(receiving)
                receiving_fnspace = receiving_super.function_space()
                receiving_identity = utils.get_identity_block(receiving_fnspace)
                receiving_idx = get_super_idx(receiving)

                rhs = FunctionAssignerRHS(self, self.adj_function_assigner, receiving_super, receiving_idx, giving)
                receiving_dep = adjglobals.adj_variables.next(receiving_super)

                solving.register_initial_conditions(zip(rhs.coefficients(),rhs.dependencies()), linear=True)
                if backend.parameters["adjoint"]["record_all"]:
                    adjglobals.adjointer.record_variable(receiving_dep, libadjoint.MemoryStorage(adjlinalg.Vector(receiving_super)))

                eq = libadjoint.Equation(receiving_dep, blocks=[receiving_identity], targets=[receiving_dep], rhs=rhs)
                cs = adjglobals.adjointer.register_equation(eq)

                solving.do_checkpoint(cs, receiving_dep, rhs)

            return out

    class FunctionAssignerRHS(libadjoint.RHS):
        def __init__(self, function_assigner, adj_function_assigner, receiving_super, receiving_idx, giving):
            self.function_assigner = function_assigner
            self.adj_function_assigner = adj_function_assigner

            self.receiving_super = receiving_super
            self.receiving_idx   = receiving_idx
            self.receiving_dep   = adjglobals.adj_variables[receiving_super]

            self.giving_list = isinstance(giving, list)
            if not self.giving_list:
                giving = [giving]

            self.giving_supers   = [get_super_function(giver) for giver in giving]
            self.giving_idxs     = [get_super_idx(giver) for giver in giving]
            self.giving_deps     = [adjglobals.adj_variables[giver] for giver in self.giving_supers]

        def __call__(self, dependencies, values):
            receiving_super = values[0].data.copy(deepcopy=True) # make a copy of the OLD value of what we're assigning to
            receiving_sub = receiving_super
            for idx in self.receiving_idx:
                receiving_sub = receiving_sub.sub(idx)

            giving = [value.data for value in values[1:]]
            giving_subs = []

            for i in range(len(giving)):
                giver = giving[i]
                for idx in self.giving_idxs[i]:
                    giver = giver.sub(idx)

                giving_subs.append(giver)

            if not self.giving_list:
                assert len(giving_subs) == 1
                giving_subs = giving_subs[0]

            self.function_assigner.assign(receiving_sub, giving_subs)
            return adjlinalg.Vector(receiving_super)

        def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
            if not hermitian:
                # FunctionAssigner.assign is linear, which makes the tangent linearisation equivalent to
                # just calling it with the right args.

                new_values = []

                for (dep, value) in zip(dependencies, values):
                    if dep == variable:
                        assert len(contraction_vector.data.vector()) == len(value.data.vector())
                        new_values.append(contraction_vector)
                    else:
                        new_values.append(value.duplicate())

                return self.__call__(dependencies, new_values)
            else:
                # The adjoint of a pack is a split. The hard part is deciding what to split, exactly!

                # Sometimes dolfin overloads too much, and does too many different things in one routine.

                # Terminology: background is the value of the function before FunctionAssigner.assignment; e.g., if we have
                # FunctionAssigner.assign(v.sub(0), u)
                # then v's previous value (the one that supplies .sub(1), .sub(2), etc after assignment) is the background
                # value.
                # Here, u is the giving function.

                # We can EITHER be assigning to a subfunction of a function, OR assigning multiple components to a mixed function.
                # i.e. it's either
                # assign(z.sub(0), u)
                # or
                # assign(z, [u, p])
                # We need to treat their adjoints differently.

                # First, let's check if we're differentiating with respect to one of the giving functions.
                for giving_dep in self.giving_deps:
                    if variable == giving_dep:
                        # Now, are we assigning all components to a mixed function, or one component to a subfunction?
                        if self.giving_list:
                            # We need to figure out the index of this component in order to decide what to split.
                            idx = giving_deps.index(giving_dep)
                            out = backend.Function(contraction_vector.data.sub(idx))
                            return adjlinalg.Vector(out)

                        # OR, we were assigning to a subfunction, in which case the index
                        # information is contained in giving_idxs and receiving_idx.
                        out = backend.Function(self.giving_supers[0].function_space())
                        out_sub = out
                        for idx in self.giving_idxs[0]:
                            out_sub = out_sub.sub(idx)

                        in_ = contraction_vector.data
                        for idx in self.receiving_idx:
                            in_ = in_.sub(idx)

                        self.adj_function_assigner.assign(out_sub, in_)

                        return adjlinalg.Vector(out)

                # If we got to here, we're differentiating with respect to the
                # background (receiving) value.
                else:
                    assert variable == self.receiving_dep

                    # Here, the derivative is the identity for all components EXCEPT that which we've changed,
                    # where it's zero.
                    # If we've changed all components, the derivative is zero:
                    if self.giving_list:
                        return adjlinalg.Vector(None)

                    # So we can call ourself with the right values, and we should get the right effect.
                    out = contraction_vector
                    new_values = [out]

                    for giving in self.giving_supers:
                        zero_giver = backend.Function(giving.function_space())
                        new_values.append(adjlinalg.Vector(zero_giver))

                    return self.__call__(dependencies, new_values)

        def dependencies(self):
            # This depends on the OLD value of the receiving function -- why?
            # Because: if we only update v.sub(0), v_new's other components
            # 1, 2, 3, ... are taken from v_old.
            return [self.receiving_dep] + self.giving_deps

        def coefficients(self):
            return [self.receiving_super] + self.giving_supers

        def __str__(self):
            return "FunctionAssignerRHS"

    def get_super_function(receiving):
        out = receiving
        while hasattr(out, 'super_fn'):
            out = out.super_fn
        return out

    def get_super_idx(receiving):
        indices = []
        sup = receiving
        while hasattr(sup, 'super_fn'):
            indices.append(sup.super_idx)
            sup = sup.super_fn

        return list(reversed(indices))
