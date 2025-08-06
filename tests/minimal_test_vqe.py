from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_algorithms import NumPyMinimumEigensolver

# Define H₂ molecule
driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)

# Build problem
problem = driver.run()

# Solve exactly
mapper = JordanWignerMapper()
solver = NumPyMinimumEigensolver()
calc = GroundStateEigensolver(mapper, solver)
result = calc.solve(problem)

print("Ground-state energy (Hartree):", result.groundenergy)
