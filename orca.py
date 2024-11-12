import os
import subprocess
import json
import re
import numpy as np
from pathlib import Path
import socket

from molecule import Molecule


class ORCA_input():
    """Class to generate ORCA input files from a configuration dictionary."""
    
    # Define constants for various settings
    VALID_SCF = {
        "loose": "LooseScf",
        "tight": "TightScf",
        "verytight": "VeryTightScf",
        None: ""
    }
    
    VALID_OPT = {
        "loose": "LooseOpt",
        "tight": "TightOpt",
        "verytight": "VeryTightOpt",
        None: ""
    }
    
    VALID_TYPES = {
        "sp": "energy",               # Single point
        "opt": "opt",                 # Geometry optimization
        "copt": "copt",               # Geometry optimization (cartesian)
        "grad": "engrad",             # Energy gradient
        "numgrad": "numgrad",         # Numerical gradient
        "freq": "freq",               # Frequency calculation
        "numfreq": "numfreq",         # Numerical frequency
        "optfreq": "opt freq",        # Optimization + Frequencies
        "ts": "optts",                # Transition state search

        "scan": "Scan",               # Coordinate scan
        "goat": "Goat"                # Conformer search
    }
    
    def __init__(self, config):
        """Initialize with configuration dictionary."""
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate the configuration dictionary."""
        required_keys = ["type", "method", "basis", "charge", "multiplicity"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
                
        # Validate calculation type
        if self.config["type"].lower() not in self.VALID_TYPES:
            raise ValueError(f"Invalid calculation type: {self.config['type']}")
            
        # Validate SCF convergence if present
        if "scf" in self.config and self.config["scf"].lower() not in self.VALID_SCF:
            raise ValueError(f"Invalid SCF convergence: {self.config['scf']}")
            
        # Validate optimization convergence if present
        if "opt" in self.config and self.config["opt"].lower() not in self.VALID_OPT:
            raise ValueError(f"Invalid optimization convergence: {self.config['opt']}")

    def _generate_header(self):
        """Generate the main ORCA command line."""
        parts = ["!"]
        
        # Add method and basis set
        parts.append(self.config["method"])
        parts.append(self.config["basis"])
        
        # Add calculation type
        calc_type = self.VALID_TYPES[self.config["type"].lower()]
        if calc_type:
            parts.append(calc_type)
        
        # Add SCF convergence criteria if specified
        if "scf" in self.config:
            scf = self.VALID_SCF[self.config["scf"].lower()]
            if scf:
                parts.append(scf)
        
        # Add optimization convergence criteria if specified
        if "opt" in self.config:
            opt = self.VALID_OPT[self.config["opt"].lower()]
            if opt:
                parts.append(opt)
                
        return " ".join(parts)

    def _generate_blocks(self):
        """Generate the % blocks for parallel execution and memory settings."""
        blocks = []

        # Base name block
        if "base" in self.config:
            blocks.append(f'%base\n         "{self.config["base"]}"')
        
        # Parallel execution block
        if "nprocs" in self.config:
            blocks.append(f"%pal\n          nprocs {self.config['nprocs']}\nend")
        
        # Memory block
        if "mem_per_proc" in self.config:
            blocks.append(f"%maxcore\n          {self.config['mem_per_proc']}")
            
        return "\n".join(blocks)


    def _generate_xyz_block(self, xyz_file=None, molecule=None):
        """Generate the xyz coordinate block."""
        charge = self.config["charge"]
        multiplicity = self.config["multiplicity"]
        
        if molecule is not None:
            coords = "\n".join(f"{symbol} {x:10.5f} {y:10.5f} {z:10.5f}" for symbol, (x, y, z) in zip(molecule.atoms, molecule.coordinates))
            return f"* xyz {charge} {multiplicity}\n{coords}\n*"
    
        elif xyz_file is not None:
            # Verify the xyz file exists
            if not os.path.exists(xyz_file):
                raise FileNotFoundError(f"XYZ file not found: {xyz_file}")
            return f"* xyzfile {charge} {multiplicity} {xyz_file} "

        else:
            return f"* xyz {charge} {multiplicity}\n\n*"

    def generate_input(self, xyz_file=None, molecule=None):
        """Generate the complete ORCA input file content."""
        parts = [
            self._generate_header(),
            "",  # Empty line after header
            self._generate_blocks(),
            "",  # Empty line before coordinates
            self._generate_xyz_block(xyz_file=xyz_file, molecule=molecule),
            ""  # Final newline
        ]
        
        return "\n".join(filter(bool, parts))  # filter out empty strings

    def write_input(self, filename, xyz_file=None, molecule=None):
        """Write the ORCA input to a file."""
        input_content = self.generate_input(xyz_file=xyz_file, molecule=molecule)
        with open(filename, 'w') as f:
            f.write(input_content)



class ORCA_output():
    """Class to handle ORCA properties file parsing with improved organization and error handling."""

    # Constants for regex patterns
    TYPE_PATTERN = r'&Type\s*"([^"]+)"'
    DIM_PATTERN = r'&Dim\s*\((\d+),(\d+)\)'
    VALUE_PATTERN = r'\]\s*([^"]*)'
    VALUE_PATTERN_STRING = r'\]\s*(.+)$'
    
    @staticmethod
    def get_data_type(type_info):
        """Extract data type from type information string."""
        match = re.search(ORCA_output.TYPE_PATTERN, type_info)
        return match.group(1) if match else None

    @staticmethod
    def get_array_info(info):
        """Extract array dimensions from information string."""
        match = re.search(ORCA_output.DIM_PATTERN, info)
        return (int(match.group(1)), int(match.group(2))) if match else None

    @staticmethod
    def convert_value(value, data_type):
        """Convert string value to appropriate Python type."""
        value = value.strip()
        if not value:
            return None
            
        converters = {
            "String": lambda x: x.strip('"'),
            "Integer": int,
            "Double": float,
            "Boolean": lambda x: x.lower() == "true",
            "ArrayOfIntegers": lambda x: [int(i) for i in x.split()[1:] if i.strip()],
            "ArrayOfDoubles": lambda x: [float(i) for i in x.split()[1:] if i.strip()],
            "Coordinates": lambda x: [str(i) for i in x.split() if i.strip()]
        }
        
        return converters.get(data_type, lambda x: x)(value)

    def parse_property_line(self, line, current_data):
        """Parse a property line starting with &."""
        parts = line.split(None, 1)
        key = parts[0][1:]
        
        if len(parts) <= 1:
            return False, None, None, None
            
        value_info = parts[1]
        data_type = self.get_data_type(value_info)
        
        if not data_type:
            current_data[key] = value_info.strip()
            return False, None, None, None
            
        array_dims = self.get_array_info(value_info)
        if array_dims:
            return True, key, array_dims, data_type
            
        # Extract value based on data type
        pattern = ORCA_output.VALUE_PATTERN if data_type in ("Double", "Integer") else ORCA_output.VALUE_PATTERN_STRING
        value_match = re.search(pattern, value_info)
        
        if value_match:
            current_data[key] = self.convert_value(value_match.group(1), data_type)
            
        return False, None, None, None

    def parse_orca_output(self, content):
        """Parse ORCA output file content into a structured dictionary."""
        # Clean and split into lines
        lines = [line.strip() for line in content.split('\n') 
                if line.strip() and not line.startswith('#')]
        
        # Split into geometry blocks
        geom_blocks = []
        current_block = []
        for line in lines:
            if any(marker in line for marker in ("$Geometry", "$Calculation_Status")):
                if current_block:
                    geom_blocks.append(current_block)
                    current_block = []
            current_block.append(line)
        if current_block:
            geom_blocks.append(current_block)
        
        result = {"Properties": []}
        
        for block in geom_blocks:
            geom_result = {}
            current_block = None
            current_data = {}
            collecting_array = False
            array_data = []
            array_key = None
            array_dims = None
            
            for line in block:
                if line.startswith('$') and not line.startswith('$End'):
                    # Start new block
                    current_block = line[1:]
                    current_data = {}
                    geom_result[current_block] = current_data
                    collecting_array = False
                    
                elif line.startswith('$End'):
                    # End block
                    if collecting_array and array_dims:
                        current_data[array_key] = array_data[-(array_dims[0]*array_dims[1]):]
                    collecting_array = False
                    
                elif line.startswith('&'):
                    # Handle property line
                    if collecting_array and array_dims:
                        current_data[array_key] = array_data[-(array_dims[0]*array_dims[1]):]
                        
                    collecting_array, array_key, array_dims, data_type = self.parse_property_line(line, current_data)
                    array_data = []
                    
                elif collecting_array:
                    # Collect array data
                    array_data.extend(self.convert_value(line, data_type))
            
            if geom_result:
                result["Properties"].append(geom_result)
        
        return result


class ORCA:
    """Class to manage ORCA calculations: input generation, execution, and output parsing."""
    
    def __init__(self, config, orca_cmd="/home/kreimendahl/software/orca_6.0.1/orca", work_dir=None):
        """
        Initialize ORCA manager.
        
        Args:
            config: Dictionary containing calculation parameters 
            work_dir: Working directory for the calculation (default: current directory)
            orca_cmd: Path to ORCA executable (default: "orca")
        """
            
        self.config = config
        self.work_dir = Path.cwd().resolve() if work_dir is None else Path(work_dir).resolve()
        self.orca_cmd = orca_cmd
        
        # Create working directory if it doesn't exist
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file paths using base name from config
        self.base_name = self.config['base'] if 'base' in self.config else "orca"
        self.input_file = None
        self.output_file = None
        self.property_file = None
        self.results = None
        
    def prepare_input(self, xyz_file=None, molecule=None):
        """Generate ORCA input file."""
        self.input_file = self.work_dir / f"{self.base_name}.inp"
        self.output_file = self.work_dir / f"{self.base_name}.out"
        self.property_file = self.work_dir / f"{self.base_name}.property.txt"
        self.xyz_file = self.work_dir / xyz_file if xyz_file else None
        
        # Generate input file
        generator = ORCA_input(self.config)
        generator.write_input(self.input_file, xyz_file=xyz_file, molecule=molecule)
        
        return self.input_file
        
    def run(self):
        """
        Execute ORCA calculation and wait for it to complete.
        
        Returns:
            subprocess.CompletedProcess
        """
        if not self.input_file:
            raise ValueError("Input file not prepared. Call prepare_input() first.")

        print(f"Running ORCA in {self.work_dir} on {socket.gethostname()}")
            
        # Prepare command
        cmd = f"{self.orca_cmd} {self.input_file} > {self.output_file}"
        
        try:
            # Run and wait for completion
            result = subprocess.run(
                cmd,
                cwd=self.work_dir,
                shell=True,
                capture_output=True,
                check=True,
                text=True,
                errors='ignore'
            )

            return result
                
        except subprocess.CalledProcessError as e:
            print(f"ORCA calculation failed with error:\n{e.stderr}")
            raise
        except Exception as e:
            print(f"Error running ORCA: {str(e)}")
            raise
            
    def check_status(self):
        """Check if the calculation has completed and was successful."""
        if not self.output_file.exists():
            return False
            
        # Check last line of output file for completion
        with open(self.output_file, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return False

        if any("ORCA TERMINATED NORMALLY" in line for line in lines[-5:]):
            return True
        else:
            print(lines)
            return False
    
    def parse_output(self):
        """Parse ORCA output and property files."""
        if not self.check_status():
            raise RuntimeError("Calculation not complete or failed.")
            
        parser = ORCA_output()
        
        # Parse property file if it exists
        if self.property_file.exists():
            with open(self.property_file, 'r') as f:
                self.results = parser.parse_orca_output(f.read())
        else:
            print("Warning: Property file not found.")
            self.results = None
            
        return self.results
    
    def clean_up(self, keep_main_files=True):
        """
        Clean up calculation files.
        
        Args:
            keep_main_files: If True, keep input, output, and property files
        """

        patterns_to_keep = ["*.inp", "*.out", "*.property.txt", "*.xyz"] if keep_main_files else []

        for file in self.work_dir.iterdir():
            if not any(file.match(pattern) for pattern in patterns_to_keep):
                file.unlink()


# Example usage
if __name__ == "__main__":

    # Check if the current hostname is not wuxcs
    assert socket.gethostname() != "wuxcs", "This script should not be run on the wuxcs."
    
    # Example configuration
    config = {
        "base": "orca",
        "type": "optfreq",
        "method": "b3lyp",
        "basis": "6-31g",
        "charge": "0",
        "multiplicity": "1",
        "nprocs": "20",
        "mem_per_proc": "5000"
    }

    # Read molecule from xyz file
    mol = Molecule().read_from_xyz("./test/n-butane.xyz")[-1]
    
    # Create ORCA manager
    orca = ORCA(config, work_dir="/scratch/2328635/")

    # Prepare input and run calculation
    orca.prepare_input(molecule=mol)
    orca.run()
    
    # Parse results (raw data)
    results = orca.parse_output()
    #print(json.dumps(results, indent=2))
    
    # Clean up temporary files
    orca.clean_up()

    # Write optimized geometry to xyz file
    Molecule().read_from_ORCA(results)[-1].write_to_xyz("opt.xyz")