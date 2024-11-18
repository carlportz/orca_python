import os
import subprocess
import json
import re
import numpy as np
from pathlib import Path
import socket

from ase.io import read


class CREST_input():
    """Class to generate Crest input files from a configuration dictionary."""
    
    # Valid Crest calculation types
    VALID_TYPES = {
        "conf": "",                   # Conformer search
        "nci": "--nci",               # Conformer search with non-covalent interactions
    }

    # Valid Crest methods
    VALID_METHODS = {
        "gfn1": "--gfn 1",
        "gfn2": "--gfn 2",
        "gfnff": "--gfnff",
    }
    
    def __init__(self, config):
        """Initialize with configuration dictionary."""
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate the configuration dictionary."""
        required_keys = ["type"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate calculation type
        if self.config["type"].lower() not in self.VALID_TYPES:
            raise ValueError(f"Invalid calculation type: {self.config['type']}")

        # Validate method if specified
        if "method" in self.config and self.config["method"].lower() not in self.VALID_METHODS:
            raise ValueError(f"Invalid method: {self.config['method']}")

    def _generate_command(self):
        """Generate the main Crest command line options."""
        parts = [""]
        
        # Add calculation type
        calc_type = self.VALID_TYPES[self.config["type"].lower()]
        if calc_type:
            parts.append(calc_type)
        
        # Add method if specified
        if "method" in self.config:
            parts.append(self.VALID_METHODS[self.config["method"].lower()])

        # Add charge if specified
        if "charge" in self.config:
            parts.append(f"--chrg {self.config['charge']}")

        # Add Spin (multiplicity - 1) if specified
        if "multiplicity" in self.config:
            spin = int(self.config["multiplicity"]) - 1
            parts.append(f"--uhf {spin}")

        # Add solvent if specified
        if "solvent" in self.config:
            parts.append(f"--alpb {self.config['solvent']}")

        # Add number of processors if specified
        if "nprocs" in self.config:
            parts.append(f"--T {self.config['nprocs']}")

        # Add other keywords are specified
        if "keywords" in self.config:
            parts.append(self.config["keywords"])
                
        return " ".join(parts)

    def _generate_blocks(self):
        blocks = []

        # Geometry constraints block
        if "constraints" in self.config:
            blocks.append(f"$constrain")
            if "force constant" in self.config:
                blocks.append(f"    force constant={self.config['force constant']}")
            for constraint in self.config["constraints"].split(';'):
                blocks.append(f"    {constraint}")
            blocks.append("$end")
        
        return "\n".join(filter(bool, blocks))  # filter out empty strings

    def generate_input(self, work_dir, xyz_file=None, molecule=None):
        """Generate the complete Crest input file, commandline options and coordinate block."""

        # Generate the main command line options
        command = self._generate_command()

        # Generate input coordinates
        input_xyz_file = work_dir / "crest_input.xyz"

        if xyz_file:
            Molecule().read_from_xyz(xyz_file)[-1].write_to_xyz(input_xyz_file)
        elif molecule:
            molecule.write_to_xyz(input_xyz_file)
        else:
            raise ValueError("No input coordinates provided.")

        # Generate the input blocks, if necessary
        if "constraints" in self.config:
            input_content = self._generate_blocks()
        else:
            input_content = None

        return command, input_xyz_file, input_content

    def write_input(self, filename, work_dir, xyz_file=None, molecule=None):
        """Write the Crest input and coordinates to a file, and extend the command line options."""
        command, input_xyz_file, input_content = self.generate_input(work_dir, xyz_file=xyz_file, molecule=molecule)

        # Write the input file
        if input_content:
            with open(filename, 'w') as f:
                f.write(input_content)
            
            command += f" --cinp {filename}"

        # Add coordinates file to the command
        command = f"{input_xyz_file}" + command

        return command


class CREST:
    """Class to manage Crest calculations: input generation, execution, and output parsing."""
    
    def __init__(self, config, crest_cmd="/home/kreimendahl/software/crest", work_dir=None):
        """
        Initialize Crest manager.
        
        Args:
            config: Dictionary containing calculation parameters 
            work_dir: Working directory for the calculation (default: current directory)
            crest_cmd: Path to Crest executable
        """
            
        self.config = config
        self.work_dir = Path.cwd().resolve() if work_dir is None else Path(work_dir).resolve()
        self.crest_cmd = crest_cmd
        
        # Create working directory if it doesn't exist
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file paths
        self.input_file = None
        self.output_file = None
        self.results = None
        
    def prepare_input(self, xyz_file=None, molecule=None):
        """Generate Crest input file, if necessary."""
        self.input_file = self.work_dir / "crest.inp"
        self.output_file = self.work_dir / "crest.out"
        self.xyz_file = Path(xyz_file).resolve() if xyz_file else None
        
        # Generate input file if necessary
        generator = CREST_input(self.config)
        self.cmd_options = generator.write_input(self.input_file, self.work_dir, xyz_file=xyz_file, molecule=molecule)
        
    def run(self):
        """
        Execute Crest calculation and wait for it to complete.
        
        Returns:
            subprocess.CompletedProcess
        """
        if not self.input_file:
            raise ValueError("Input file not prepared. Call prepare_input() first.")

        print(f"Running Crest in {self.work_dir} on {socket.gethostname()}")
            
        # Prepare command
        cmd = f"{self.crest_cmd} {self.cmd_options} > {self.output_file}"
        print(f"Running command: {cmd}")
        
        try:
            # Run and wait for completion
            self.cmd_result = subprocess.run(
                cmd,
                cwd=self.work_dir,
                shell=True,
                capture_output=True,
                check=True,
                text=True,
                errors='ignore'
            )
                
        except subprocess.CalledProcessError as e:
            print(f"Crest calculation failed with error:\n{e.stderr}")
            raise
        except Exception as e:
            print(f"Error running Crest: {str(e)}")
            raise

        # Parse output
        self.results = self.parse_output()

        # Add configuration to results
        self.results["Configuration"] = self.config

        # Clean up temporary files
        self.clean_up()
            
    def check_status(self):
        """Check if the calculation has completed and was successful."""
        if not self.output_file.exists():
            return False
            
        # Check last line of output file for completion
        with open(self.output_file, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return False

        if any("CREST terminated normally." in line for line in lines[-5:]):
            return True
        else:
            return False
            
    def parse_output(self):
        """Parse Crest energy file."""
        if not self.check_status():
            raise RuntimeError("Calculation not complete or failed.")
            
        # Parse energy file if it exists
        energy_file = self.work_dir / "crest.energies"
        if energy_file.exists():
            energies = np.loadtxt(energy_file, usecols=(1,)).tolist()
        else:
            print("Warning: Energy file not found.")
            energies = None
            
        return {"energies": energies}
    
    def clean_up(self, keep_main_files=True):
        """
        Clean up calculation files.
        
        Args:
            keep_main_files: If True, keep input, output, and property files
        """

        patterns_to_keep = ["*.inp", "*.out", "*.xyz", "*.coord"] if keep_main_files else []

        for file in self.work_dir.iterdir():
            if not any(file.match(pattern) for pattern in patterns_to_keep):
                file.unlink()

    def get_ensemble(self):
        """Return optimized ensemble from XTB output as ASE Atoms object."""
        if not self.results:
            raise ValueError("No results available. Run calculation first.")
        
        # Construct path to last geometry
        mol_path = self.work_dir / "crest_conformers.xyz"
        if not mol_path.exists():
            raise FileNotFoundError(f"Ensemble file not found: {mol_path}")

        # Read last geometry from file
        ensemble = read(mol_path, format="xyz", index=":")
        
        return ensemble


# Example usage
if __name__ == "__main__":

    # Check if the current hostname is not wuxcs
    assert socket.gethostname() != "wuxcs", "This script should not be run on the wuxcs."
    
    # Example configuration
    config = {
        "type": "conf",
        #"method": "gfn2",
        "nprocs": "20",
        #"constraints": "dihedral: 1,2,3,4,90.0",
        "constraints": "angle: 2,3,4,180.0",
        "force constant": "0.25",
        #"solvent": "water",
    }

    # Read molecule from xyz file
    mol = Molecule().read_from_xyz("./test/imine_52.xyz")[-1]
    
    # Create CREST manager
    crest = CREST(config, work_dir="/scratch/2328635/")

    # Prepare input and run calculation
    crest.prepare_input(molecule=mol)
    crest.run()
    
    # Parse results (raw data)
    print(json.dumps(crest.results, indent=2))

