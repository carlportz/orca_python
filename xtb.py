import os
import subprocess
import json
import re
import numpy as np
from pathlib import Path
import socket

from ase import Atoms
from ase.io import read, write

class XTB_input():
    """Class to generate XTB input files from a configuration dictionary."""
    
    # Valid XTB calculation types
    VALID_TYPES = {
        "sp": "--scc",                # Single point
        "opt": "--opt",               # Geometry optimization
        "grad": "--grad",             # Energy gradient
        "freq": "--hess",             # Frequency calculation
        "optfreq": "--ohess",         # Optimization + Frequencies
        "md": "--md",                 # Molecular dynamics
        "metadyn": "--metadyn",       # Metadynamics
    }

    # Valid XTB methods
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
        """Generate the main XTB command line options."""
        parts = [""]
        
        # Add calculation type
        calc_type = self.VALID_TYPES[self.config["type"].lower()]
        if calc_type:
            parts.append(calc_type)

        # Add base name if specified
        if "base" in self.config:
            parts.append(f"--namespace {self.config['base']}")
        
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
            parts.append(f"--parallel {self.config['nprocs']}")

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

        # Relaxed scan block
        if "scan" in self.config:
            blocks.append(f"$scan")
            for scan in self.config["scan"].split(';'):
                blocks.append(f"    {scan}")
            blocks.append("$end")
        
        return "\n".join(filter(bool, blocks))  # filter out empty strings

    def generate_input(self, work_dir, molecule=None):
        """Generate the complete XTB input file, commandline options and coordinate block."""

        # Generate the main command line options
        command = self._generate_command()

        # Generate the input blocks, if necessary
        if "constraints" in self.config or "scan" in self.config:
            input_content = self._generate_blocks()
        else:
            input_content = None

        return command, input_content

    def write_input(self, filename, work_dir, molecule=None):
        """Write the XTB input and coordinates to a file, and extend the command line options."""
        command, input_content = self.generate_input(work_dir, molecule=molecule)

        # Write the input file
        if input_content:
            with open(filename, 'w') as f:
                f.write(input_content)
            
            command += f" --input {filename}"

        # Write the coordinates to a separate file
        coords_file = work_dir / f"{self.config['base']}_input.xyz"
        write(coords_file, molecule, format='xyz')

        # Add coordinates file to the command
        command += f" -- {coords_file}"

        return command


class XTB_output():
    """Class to handle XTB output file parsing."""

    def _find_property_section(self, content):
        """Find the section in the output file containing property data."""
        lines = content.split('\n')
        start_idx = -1
        
        for i, line in reversed(list(enumerate(lines))):
            if "SUMMARY" in line:
                start_idx = i
                break
                    
        if start_idx == -1:
            raise ValueError("Could not find SUMMARY in file content")
            
        return start_idx

    def parse_xtb_output(self, content):
        """Parse XTB output file SUMMARY into a structured dictionary."""
        result = {}   
        start_idx = self._find_property_section(content)
        lines = content.split('\n')
        
        # Process lines between dashes
        for line in lines[start_idx+2:]:
            line = line.strip()
            if ':' in line:
                # Remove leading/trailing ':' and split by whitespace
                clean_line = line.replace(':', '').strip()
                parts = clean_line.split()
                
                if len(parts) >= 2:
                    # Join all parts except the last two (value and unit) as the key
                    key = ' '.join(parts[:-2]).strip()
                    value = float(parts[-2])  # Convert value to float
                    result[key] = value     
            
            if '..........' in line:
                break
        
        return result


class XTB:
    """Class to manage XTB calculations: input generation, execution, and output parsing."""
    
    def __init__(self, config, xtb_cmd="/home/kreimendahl/software/orca_6.0.1/otool_xtb", work_dir=None):
        """
        Initialize XTB manager.
        
        Args:
            config: Dictionary containing calculation parameters 
            work_dir: Working directory for the calculation (default: current directory)
            xtb_cmd: Path to xtb executable
        """
            
        self.config = config
        self.work_dir = Path.cwd().resolve() if work_dir is None else Path(work_dir).resolve()
        self.xtb_cmd = xtb_cmd
        
        # Create working directory if it doesn't exist
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file paths using base name from config
        self.base_name = self.config['base'] if 'base' in self.config else "orca"
        self.input_file = None
        self.output_file = None
        self.results = None
        
    def prepare_input(self, molecule=None):
        """Generate XTB input file, if necessary."""
        self.input_file = self.work_dir / f"{self.base_name}.inp"
        self.output_file = self.work_dir / f"{self.base_name}.out"
        
        # Generate input file if necessary
        generator = XTB_input(self.config)
        self.cmd_options = generator.write_input(self.input_file, self.work_dir, molecule=molecule)
        
    def run(self):
        """
        Execute XTB calculation and wait for it to complete.
        
        Returns:
            subprocess.CompletedProcess
        """
        if not self.input_file:
            raise ValueError("Input file not prepared. Call prepare_input() first.")

        print(f"Running XTB in {self.work_dir} on {socket.gethostname()}")
            
        # Prepare command
        cmd = f"{self.xtb_cmd} {self.cmd_options} > {self.output_file}"
        
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
            print(f"XTB calculation failed with error:\n{e.stderr}")
            raise
        except Exception as e:
            print(f"Error running XTB: {str(e)}")
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
        else:
            return True
            
    def parse_output(self):
        """Parse XTB output file."""
        if not self.check_status():
            raise RuntimeError("Calculation not complete or failed.")
            
        parser = XTB_output()
        
        # Parse output file if it exists
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                self.results = parser.parse_xtb_output(f.read())
        else:
            print("Warning: Output file not found.")
            self.results = None
            
        return self.results
    
    def clean_up(self, keep_main_files=True):
        """
        Clean up calculation files.
        
        Args:
            keep_main_files: If True, keep input, output, and property files
        """

        patterns_to_keep = ["*.inp", "*.out", "*.xyz", "*.coord", "*.log"] if keep_main_files else []

        for file in self.work_dir.iterdir():
            if not any(file.match(pattern) for pattern in patterns_to_keep):
                file.unlink()

    def get_molecule(self):
        """Return last molecule from XTB output as ASE Atoms object."""
        if not self.results:
            raise ValueError("No results available. Run calculation first.")
        
        # Construct path to last geometry
        mol_path = self.work_dir / f"{self.base_name}.xtbopt.xyz"
        if not mol_path.exists():
            raise FileNotFoundError(f"Geometry file not found: {mol_path}")

        # Read last geometry from file
        mol = read(mol_path, format="xyz")
        
        return mol


# Example usage
if __name__ == "__main__":

    # Check if the current hostname is not wuxcs
    assert socket.gethostname() != "wuxcs", "This script should not be run on the wuxcs."
    
    # Example configuration
    config = {
        "base": "test",
        "type": "optfreq",
        "method": "gfn2",
        "charge": "0",
        "multiplicity": "1",
        "nprocs": "2",
        "constraints": "dihedral: 1,2,3,4,90.0",
        #"scan": "D 0 1 2 3 = 180.0, 0.0, 3",
        "solvent": "water",
    }

    # Read molecule from xyz file
    mol = read("./test/water.xyz", format='xyz')
    
    # Create XTB manager
    xtb = XTB(config, work_dir="/scratch/2328635/")

    # Prepare input and run calculation
    xtb.prepare_input(molecule=mol)
    xtb.run()
    
    # Print results
    print(json.dumps(xtb.results, indent=2))
