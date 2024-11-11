import numpy as np
import json
from pathlib import Path

class Molecule:
    """Class to handle molecular structure and basic file I/O."""
    
    def __init__(self, atoms=None, coordinates=None, charge=0, multiplicity=1, context=None):
        """
        Initialize Molecule.
        
        Args:
            atoms: List of atomic symbols
            coordinates: List/array of atomic coordinates (Nx3, Angstrom)
            charge: Molecular charge
            multiplicity: Spin multiplicity
            context: Dictionary with additional information (optional)
        """
        self.atoms = atoms if atoms is not None else []
        self.coordinates = np.array(coordinates) if coordinates is not None else np.array([])
        self.charge = charge
        self.multiplicity = multiplicity
        self.context = context if context is not None else {}
        
        # Validate if both atoms and coordinates are provided
        if len(self.atoms) != len(self.coordinates) and \
           (len(self.atoms) > 0 or len(self.coordinates) > 0):
            raise ValueError("Number of atoms must match number of coordinates")
    
    @property
    def n_atoms(self):
        """Return number of atoms."""
        return len(self.atoms)
        
    def read_from_xyz(self, filename):
        """
        Read multiple molecular structures from an XYZ file that contains stacked geometries.
        Each geometry can have a different number of atoms.
        
        Args:
            filename: Path to xyz file
        Returns:
            list[Molecule]: List of Molecule objects, one for each geometry in the file
        """
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"XYZ file not found: {filename}")
            
        molecules = []
        
        try:
            with open(filepath, 'r') as f:
                while True:
                    # Try to read number of atoms
                    n_atoms_line = f.readline().strip()
                    if not n_atoms_line:  # End of file
                        break
                        
                    n_atoms = int(n_atoms_line)
                    
                    # Skip comment line
                    f.readline()
                    
                    # Read atomic coordinates for this geometry
                    atoms = []
                    coordinates = []
                    
                    for _ in range(n_atoms):
                        line = f.readline().strip()
                        if not line:
                            raise ValueError("Unexpected end of file while reading coordinates")
                        
                        parts = line.split()
                        if len(parts) < 4:
                            raise ValueError(f"Invalid coordinate line: {line}")
                        
                        atom = parts[0]
                        coords = [float(x) for x in parts[1:4]]
                        
                        atoms.append(atom)
                        coordinates.append(coords)
                    
                    # Create molecule object for this geometry and add to list
                    molecules.append(Molecule(atoms=atoms, coordinates=np.array(coordinates)))
                    
        except Exception as e:
            raise ValueError(f"Error parsing XYZ file: {str(e)}")

        if not molecules:
            raise ValueError("No valid molecular geometries found in file")
            
        return molecules    


    def write_xyz(self, filename):
        """
        Write molecular structure to xyz file.
        Context dictionary is written as JSON in the comment line.
        
        Args:
            filename: Output filename
        """
        if len(self.atoms) == 0:
            raise ValueError("No molecular structure to write")
            
        with open(filename, 'w') as f:
            f.write(f"{self.n_atoms}\n")
            # Write context as JSON in comment line
            f.write(f"{json.dumps(self.context)}\n")
            
            for atom, coord in zip(self.atoms, self.coordinates):
                f.write(f"{atom:2s} {coord[0]:15.10f} {coord[1]:15.10f} {coord[2]:15.10f}\n")
    
    def __str__(self):
        """String representation of the molecule."""
        lines = [f"{atom:2s} {coord[0]:10.5f} {coord[1]:10.5f} {coord[2]:10.5f}" 
                 for atom, coord in zip(self.atoms, self.coordinates)]
        context_str = f", context={self.context}" if self.context else ""
        return (f"Molecule with {self.n_atoms} atoms, charge={self.charge}, "
                f"multiplicity={self.multiplicity}{context_str}\n" + "\n".join(lines))
    
    def copy(self):
        """Return a deep copy of the molecule."""
        return Molecule(
            atoms=self.atoms.copy(),
            coordinates=self.coordinates.copy(),
            charge=self.charge,
            multiplicity=self.multiplicity,
            context=self.context.copy()
        )

    def read_from_ORCA(self, results):
        """
        Create a list of new molecules from a results dictionary from ORCA calculations.
        
        Args:
            results: Dictionary from ORCA properties file
        """

        BOHR_TO_ANGSTROM = 0.5291772105  # Conversion factor for Bohr to Angstrom
        
        molecules = []
        for mol in results['Properties']:
            if 'Geometry' not in mol:
                continue
            
            natoms = mol['Geometry']['NATOMS']
            if natoms <= 0:
                raise ValueError(f"Invalid number of atoms: {natoms}")
            
            atoms = []
            coordinates = []
            coord_data = mol['Geometry']['CartesianCoordinates']
            for i in range(natoms):
                try:
                    atom = coord_data[i * 4]
                    x = float(coord_data[i * 4 + 1]) * BOHR_TO_ANGSTROM
                    y = float(coord_data[i * 4 + 2]) * BOHR_TO_ANGSTROM
                    z = float(coord_data[i * 4 + 3]) * BOHR_TO_ANGSTROM
                except (IndexError, ValueError) as e:
                    raise ValueError(f"Error parsing Cartesian coordinates: {str(e)}")
            
                atoms.append(atom)
                coordinates.append([x, y, z])
            
            molecules.append(Molecule(atoms=atoms, coordinates=coordinates))
        
        return molecules


# Example usage
if __name__ == "__main__":
    # Create a water molecule with context
    water = Molecule(
        atoms=['O', 'H', 'H'],
        coordinates=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9572],
            [0.9239, 0.0, -0.2399]
        ],
        charge=0,
        multiplicity=1,
        context={
            "name": "water",
            "energy": -76.4,
            "method": "B3LYP/def2-SVP"
        }
    )
    
    # Write to xyz file
    water.write_xyz("water.xyz")
    
    # Create new molecule and read the xyz file
    mol = Molecule().read_from_xyz("water.xyz")
    print(mol[-1])