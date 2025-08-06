
import sys
import os
import pytest

from qiskit_nature.second_q.drivers import PySCFDriver
#from qiskit_nature.second_q.drivers import ElectronicStructureDriverResult

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from VQEBase import AtomDriver


def test_atom_driver_initialization():
    """Test that AtomDriver initializes correctly and inherits PySCFDriver."""
    driver = AtomDriver(atom="H 0 0 0; H 0 0 1", charge=0, spin=0, basis="sto3g")
    
    # Check type inheritance
    assert isinstance(driver, PySCFDriver), "AtomDriver should inherit from PySCFDriver"
    
    # Check attributes
    assert driver.basis == "sto3g"



@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_atom_driver_run():
    """Test that AtomDriver.run() executes and returns a valid result."""
    driver = AtomDriver(atom="H 0 0 0; H 0 0 1", charge=0, spin=0, basis="sto3g")
    result = driver.run()
    
    # Verify returned object type
    #assert isinstance(result, ElectronicStructureDriverResult), \
    #    "run() should return an ElectronicStructureDriverResult"
    
    # Ensure the number of molecular orbitals > 0
    assert result.num_spatial_orbitals > 0, \
        "Result should contain at least one molecular orbital"

    assert result.molecule.charge == 0
    assert result.molecule.multiplicity == 1


if __name__ == "__main__":
    pytest.main([__file__])
