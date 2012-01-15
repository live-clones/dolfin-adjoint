import libadjoint
from solving import *

def replay_dolfin(forget=False):
  if "record_all" not in debugging or debugging["record_all"] is not True:
    print "Warning: your replay test will be much more effective with debugging['record_all'] = True."

  for i in range(adjointer.equation_count):
      (fwd_var, output) = adjointer.get_forward_solution(i)

      storage = libadjoint.MemoryStorage(output)
      storage.set_compare(tol=0.0)
      storage.set_overwrite(True)
      adjointer.record_variable(fwd_var, storage)

      if forget:
        adjointer.forget_forward_equation(i)

def convergence_order(errors):
  import math

  orders = [0.0] * (len(errors)-1)
  for i in range(len(errors)-1):
    orders[i] = math.log(errors[i]/errors[i+1], 2)

  return orders

def adjoint_dolfin(functional, forget=True):

  for i in range(adjointer.equation_count)[::-1]:
      (adj_var, output) = adjointer.get_adjoint_solution(i, functional)
      
      storage = libadjoint.MemoryStorage(output)
      adjointer.record_variable(adj_var, storage)

      if forget:
        adjointer.forget_adjoint_equation(i)

  return output.data # return the last adjoint state

def test_initial_condition_adjoint(J, ic, final_adjoint, seed=0.01, perturbation_direction=None):
  '''forward must be a function that takes in the initial condition (ic) as a dolfin.Function
     and returns the functional value by running the forward run:

       func = J(ic)

     final_adjoint is the adjoint associated with the initial condition
     (usually the last adjoint equation solved).

     This function returns the order of convergence of the Taylor
     series remainder, which should be 2 if the adjoint is working
     correctly.'''

  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  print "Running Taylor remainder convergence analysis for the adjoint model ... "
  import random
  import numpy

  # First run the problem unperturbed
  ic_copy = dolfin.Function(ic)
  f_direct = J(ic_copy)

  # Randomise the perturbation direction:
  if perturbation_direction is None:
    perturbation_direction = dolfin.Function(ic.function_space())
    vec = perturbation_direction.vector()
    for i in range(len(vec)):
      vec[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values = []
  perturbations = []
  for perturbation_size in [seed/(2**i) for i in range(5)]:
    perturbation = dolfin.Function(perturbation_direction)
    vec = perturbation.vector()
    vec *= perturbation_size
    perturbations.append(perturbation)

    perturbed_ic = dolfin.Function(ic)
    vec = perturbed_ic.vector()
    vec += perturbation.vector()

    functional_values.append(J(perturbed_ic))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

  print "Taylor remainder without adjoint information: ", no_gradient
  print "Convergence orders for Taylor remainder without adjoint information (should all be 1): ", convergence_order(no_gradient)

  adjoint_vector = numpy.array(final_adjoint.vector())

  with_gradient = []
  for i in range(len(perturbations)):
    remainder = abs(functional_values[i] - f_direct - numpy.dot(adjoint_vector, numpy.array(perturbations[i].vector())))
    with_gradient.append(remainder)

  print "Taylor remainder with adjoint information: ", with_gradient
  print "Convergence orders for Taylor remainder with adjoint information (should all be 2): ", convergence_order(with_gradient)

  return min(convergence_order(with_gradient))

def tlm_dolfin(parameter, forget=False):
  for i in range(adjointer.equation_count):
      (tlm_var, output) = adjointer.get_tlm_solution(i, parameter)

      storage = libadjoint.MemoryStorage(output)
      storage.set_overwrite(True)
      adjointer.record_variable(tlm_var, storage)

      if forget:
        adjointer.forget_forward_equation(i)
  return output

def test_initial_condition_tlm(J, dJ, ic, seed=0.01, perturbation_direction=None):
  '''forward must be a function that takes in the initial condition (ic) as a dolfin.Function
     and returns the functional value by running the forward run:

       func = J(ic)

     final_adjoint is the tangent linear variable for the solution on which the functional depends
     (usually the last TLM equation solved).

     dJ must be the derivative of the functional with respect to its argument, evaluated and assembled at
     the unperturbed solution (a dolfin Vector).

     This function returns the order of convergence of the Taylor
     series remainder, which should be 2 if the adjoint is working
     correctly.'''

  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  print "Running Taylor remainder convergence analysis for the tangent linear model... "
  import random
  import numpy

  # First run the problem unperturbed
  ic_copy = dolfin.Function(ic)
  f_direct = J(ic_copy)

  # Randomise the perturbation direction:
  if perturbation_direction is None:
    perturbation_direction = dolfin.Function(ic.function_space())
    vec = perturbation_direction.vector()
    for i in range(len(vec)):
      vec[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values = []
  perturbations = []
  for perturbation_size in [seed/(2**i) for i in range(5)]:
    perturbation = dolfin.Function(perturbation_direction)
    vec = perturbation.vector()
    vec *= perturbation_size
    perturbations.append(perturbation)

    perturbed_ic = dolfin.Function(ic)
    vec = perturbed_ic.vector()
    vec += perturbation.vector()

    functional_values.append(J(perturbed_ic))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

  print "Taylor remainder without adjoint information: ", no_gradient
  print "Convergence orders for Taylor remainder without adjoint information (should all be 1): ", convergence_order(no_gradient)

  with_gradient = []
  for i in range(len(perturbations)):
    param = InitialConditionParameter(ic, perturbations[i])
    final_tlm = tlm_dolfin(param).data
    remainder = abs(functional_values[i] - f_direct - numpy.dot(numpy.array(final_tlm.vector()), numpy.array(dJ)))
    with_gradient.append(remainder)

  print "Taylor remainder with adjoint information: ", with_gradient
  print "Convergence orders for Taylor remainder with tangent linear information (should all be 2): ", convergence_order(with_gradient)

  return min(convergence_order(with_gradient))
