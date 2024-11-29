import subprocess
import json
import re
from pathlib import Path
import shutil
import socket

from ase import Atoms
from ase.io import read

class OrcaInput:
    """
    Class to generate ORCA input files from a configuration dictionary.

    Attributes:
        config (dict): Configuration dictionary containing settings for the ORCA input file.

    Constants:
        VALID_SCF (dict): Valid SCF convergence criteria.
        VALID_OPT (dict): Valid optimization convergence criteria.
        VALID_TYPES (dict): Valid calculation types.

    Methods:
        __init__(config):
            Initialize with configuration dictionary.
        _validate_config():
            Validate the configuration dictionary.
        _generate_keyword_line():
            Generate the main ORCA command line.
        _generate_blocks():
            Generate the % blocks for parallel execution and memory settings.
        _generate_xyz_block(molecule=None):
            Generate the xyz coordinate block.
        generate_input(molecule=None):
            Generate the complete ORCA input file content.
        write_input(filename, molecule=None):
            Write the ORCA input to a file.
    """
    
    # Valid ORCA SCF convergence criteria
    VALID_SCF = {
        "loose": "LooseScf",
        "tight": "TightScf",
        "verytight": "VeryTightScf",
        None: ""
    }
    
    # Valid ORCA optimization convergence criteria
    VALID_OPT = {
        "loose": "LooseOpt",
        "tight": "TightOpt",
        "verytight": "VeryTightOpt",
        None: ""
    }
    
    # Valid ORCA calculation types
    VALID_TYPES = {
        "sp": "energy",               # Single point
        "opt": "opt",                 # Geometry optimization
        "grad": "engrad",             # Energy gradient
        "numgrad": "numgrad",         # Numerical gradient
        "freq": "freq",               # Frequency calculation
        "numfreq": "numfreq",         # Numerical frequency
        "optfreq": "opt freq",        # Optimization + Frequencies
        "ts": "optts",                # Transition state search
        "goat": "goat"                # Conformer search
    }
    
    def __init__(self, config):
        """
        Initialize the ORCA_input instance with the provided configuration.

        Args:
            config (dict): A dictionary containing configuration parameters.
        """
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """
        Validate the configuration dictionary to ensure all required keys and values are present and correct.

        Raises:
            ValueError: If any required key is missing or contains an invalid value.
        """
        required_keys = ["type", "method", "basis"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
                
        # Validate calculation type
        if self.config["type"].lower() not in self.VALID_TYPES:
            raise ValueError(f"Invalid calculation type: {self.config['type']}. Valid types are: {', '.join(self.VALID_TYPES.keys())}")
            
        # Validate SCF convergence if present
        if "scf" in self.config and self.config["scf"].lower() not in self.VALID_SCF:
            raise ValueError(f"Invalid SCF convergence: {self.config['scf']}. Valid values are: {', '.join(self.VALID_SCF.keys())}")
            
        # Validate optimization convergence if present
        if "opt" in self.config and self.config["opt"].lower() not in self.VALID_OPT:
            raise ValueError(f"Invalid optimization convergence: {self.config['opt']}. Valid values are: {', '.join(self.VALID_OPT.keys())}")

    def _generate_keyword_line(self):
        """
        Generate the main ORCA keyword line based on the configuration.

        Returns:
            str: The generated ORCA keyword line as a single string.
        """
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
        
        # Add solvent model if specified
        if "solvent" in self.config:
            parts.append(self.config["solvent"])

        # Add other keywords if specified
        if "keywords" in self.config:
            parts.append(self.config["keywords"])
                
        return " ".join(parts)

    def _generate_blocks(self):
        """
        Generate the % blocks based on the configuration.

        Returns:
            str: A string containing the concatenated blocks of settings.
        """
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

        # Geometry constraints block
        if "constraints" in self.config:
            blocks.append(f"%geom\n          constraints")
            for constraint in self.config["constraints"].split(';'):
                blocks.append(f"          {{ {constraint} }}")
            blocks.append("          end\nend")

        # Relaxed scan block
        if "scan" in self.config:
            blocks.append(f"%geom\n          scan")
            for scan in self.config["scan"].split(';'):
                blocks.append(f"          {scan}")
            blocks.append("          end\nend")

        # Excited states block
        if "nroots" in self.config:
            blocks.append(f"%tddft\n            nroots {self.config['nroots']}")
            if "iroot" in self.config:
                blocks.append(f"          iroot {self.config['iroot']}")
            blocks.append("end")

        return "\n".join(blocks)


    def _generate_xyz_block(self, molecule=None):
        """
        Generate the xyz coordinate block for a molecule.

        Args:
            molecule (ase.Atoms, optional): An ASE Atoms object representing the molecule.

        Returns:
            str: A string representing the xyz coordinate block, including charge and multiplicity, formatted for ORCA input files.
        """
        charge = self.config["charge"] if "charge" in self.config else 0
        multiplicity = self.config["multiplicity"] if "multiplicity" in self.config else 1
        
        if molecule is not None:
            coords = []
            for atom in molecule:
                x = atom.position[0]
                y = atom.position[1]
                z = atom.position[2]
                coords.append(f"{atom.symbol} {x:10.5f} {y:10.5f} {z:10.5f}")
            coords_str = "\n".join(coords)
            return f"* xyz {charge} {multiplicity}\n{coords_str}\n*"
    
        else:
            return f"* xyz {charge} {multiplicity}\n\n*"

    def generate_input(self, molecule=None):
        """
        Generate the complete ORCA input file content.

        Args:
            molecule (ase.Atoms, optional): An ASE Atoms object representing the molecule for which the ORCA input file is being generated.

        Returns:
            str: The complete ORCA input file content as a single string.
        """
        parts = [
            self._generate_keyword_line(),
            self._generate_blocks(),
            self._generate_xyz_block(molecule=molecule),
        ]
        
        return "\n".join(filter(bool, parts))  # filter out empty strings

    def write_input(self, filename, molecule=None):
        """
        Write the ORCA input to a file.

        Args:
            filename (str): The name of the file to write the input to.
            molecule (ase.Atoms, optional): The ASE Atoms object to generate the input for. If not provided, a default hydrogen atom is used.
        """
        if not molecule:
            molecule = Atoms("H", positions=[[0, 0, 0]])
            print("Warning: No molecule provided. Using default hydrogen atom.")

        input_content = self.generate_input(molecule=molecule)
        with open(filename, 'w') as f:
            f.write(input_content)


class OrcaOutput:
    """
    Class to handle ORCA properties file parsing with improved organization and error handling.

    Constants:
        TYPE_PATTERN (str): Regex pattern to match the type information.
        DIM_PATTERN (str): Regex pattern to match the array dimensions.
        VALUE_PATTERN (str): Regex pattern to match the value for numeric types.
        VALUE_PATTERN_STRING (str): Regex pattern to match the value for string types.

    Methods:
        get_data_type(type_info: str) -> Optional[str]:
            Extract the data type from the type information string.
        get_array_info(info: str) -> Optional[Tuple[int, int]]:
            Extract the array dimensions from the information string.
        convert_value(value: str, data_type: str) -> Any:
            Convert a string value to the appropriate Python type based on the data type.
        parse_property_line(line: str, current_data: dict) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]], Optional[str]]:
            Parse a property line starting with '&' and update the current data dictionary.
        parse_orca_output(content: str) -> dict:
            Parse the ORCA output file content into a structured dictionary.
    """

    # Constants for regex patterns
    TYPE_PATTERN = r'&Type\s*"([^"]+)"'
    DIM_PATTERN = r'&Dim\s*\((\d+),(\d+)\)'
    VALUE_PATTERN = r'\]\s*([^"]*)'
    VALUE_PATTERN_STRING = r'\]\s*(.+)$'
    
    @staticmethod
    def get_data_type(type_info):
        """
        Extract the data type from the type information string.

        Args:
            type_info (str): A string containing type information.

        Returns:
            str or None: The extracted data type if a match is found, otherwise None.
        """
        match = re.search(OrcaOutput.TYPE_PATTERN, type_info)

        return match.group(1) if match else None

    @staticmethod
    def get_array_info(info):
        """
        Extract array dimensions from the information string.

        Args:
            info (str): The information string containing array dimensions.

        Returns:
            tuple: A tuple containing two integers representing the array dimensions if the pattern is found, otherwise None.
        """
        match = re.search(OrcaOutput.DIM_PATTERN, info)

        return (int(match.group(1)), int(match.group(2))) if match else None

    @staticmethod
    def convert_value(value, data_type):
        """
        Convert a string value to the appropriate Python type based on the specified data type.

        Args:
            value (str): The string value to be converted.
            data_type (str): The target data type for conversion.

        Returns:
            Any: The converted value in the appropriate Python type, or None if the input value is empty.
        """
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
        """
        Parse a property line starting with '&' and update the current data dictionary.

        Args:
            line (str): The line to be parsed.
            current_data (dict): The dictionary to be updated with parsed data.

        Returns:
            tuple: A tuple containing:
                - bool: Indicates if the line represents an array.
                - str or None: The key for the property if it is an array, otherwise None.
                - list or None: The dimensions of the array if it is an array, otherwise None.
                - str or None: The data type of the property if it is an array, otherwise None.
        """
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
        pattern = OrcaOutput.VALUE_PATTERN if data_type in ("Double", "Integer") else OrcaOutput.VALUE_PATTERN_STRING
        value_match = re.search(pattern, value_info)
        
        if value_match:
            current_data[key] = self.convert_value(value_match.group(1), data_type)
            
        return False, None, None, None

    def parse_orca_output(self, content):
        """
        Parse ORCA output file content into a structured dictionary.

        Args:
            content (str): The content of the ORCA output file as a string.

        Returns:
            dict: A dictionary containing parsed data from the ORCA output file.
        """
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
                    array_data.append(self.convert_value(line, data_type))
            
            if geom_result:
                result["Properties"].append(geom_result)
        
        return result


class Orca:
    """
    Class to manage ORCA calculations: input generation, execution, and output parsing.

    Attributes:
        config (dict): Dictionary containing calculation parameters.
        work_dir (Path): Working directory for the calculation.
        orca_cmd (str): Path to ORCA executable.
        base_name (str): Base name for input, output, and property files.
        input_file (Path): Path to the ORCA input file.
        output_file (Path): Path to the ORCA output file.
        property_file (Path): Path to the ORCA property file.
        results (dict): Parsed results from the ORCA calculation.

    Constants:
        PATTERNS_TO_KEEP (list): List of file patterns to keep after cleaning

    Methods:
        prepare_input(molecule=None):
            Generate ORCA input file.
        read_input(input_file):
            Read ORCA input from an existing file.
        run():
            Execute ORCA calculation and wait for it to complete.
        check_status():
            Check if the calculation has completed and was successful.
        parse_output():
            Parse ORCA output and property files.
        clean_up(keep_main_files=True):
            Clean up calculation files.
        get_molecule():
            Return the last molecule from ORCA output as an ASE Atoms object.
    """

    # Constant for file patterns to keep
    PATTERNS_TO_KEEP = ["*.inp", 
                        "*.out", 
                        "*.property.txt", 
                        "*.xyz"]
    
    def __init__(self, config, orca_cmd=None, work_dir=None):
        """
        Initialize the ORCA class with configuration, command path, and working directory.

        Args:
            config (dict): Configuration dictionary containing necessary parameters.
            orca_cmd (str, optional): Path to the ORCA executable. Defaults to the system path.
            work_dir (str or Path, optional): Path to the working directory. Defaults to the current working directory.
        """
        self.config = config
        self.work_dir = Path.cwd().resolve() if work_dir is None else Path(work_dir).resolve()
        self.orca_cmd = shutil.which("orca") if orca_cmd is None else orca_cmd
        
        # Create working directory if it doesn't exist
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file paths using base name from config
        self.base_name = self.config['base'] if 'base' in self.config else "orca"
        self.input_file = None
        self.output_file = None
        self.property_file = None
        self.results = None
        
    def prepare_input(self, molecule=None):
        """
        Generate ORCA input file and set up file paths.

        Args:
            molecule (ase.Atoms, optional): The molecular structure to be used in the ORCA calculation. If not provided, a default structure will be used.
        """
        self.input_file = self.work_dir / f"{self.base_name}.inp"
        self.output_file = self.work_dir / f"{self.base_name}.out"
        self.property_file = self.work_dir / f"{self.base_name}.property.txt"
        
        # Generate input file
        generator = OrcaInput(self.config)
        generator.write_input(self.input_file, molecule=molecule)

    def read_input(self, input_file):
        """
        Read ORCA input from an existing input file.

        Args:
            input_file (str or Path): The path to the ORCA input file.
        """
        self.input_file = Path(input_file).resolve()
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        self.output_file = self.work_dir / f"{self.base_name}.out"
        self.property_file = self.work_dir / f"{self.base_name}.property.txt"

    def run(self):
        """
        Execute the ORCA quantum chemistry software with the prepared input file.

        Raises:
            ValueError: If the input file is not prepared.
            subprocess.CalledProcessError: If the ORCA calculation fails.
            Exception: If there is an error running ORCA.
        """
        
        if not self.input_file:
            raise ValueError("Input file not prepared. Call prepare_input() first.")

        if "verbose" in self.config and self.config["verbose"]:
            print(f"ORCA running in {self.work_dir} on {socket.gethostname()}")

        # Clean up temporary files
        self.clean_up()
        
        # Prepare command
        cmd = f"{self.orca_cmd} {self.input_file} > {self.output_file}"
        
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
            print(f"ORCA calculation failed with error:\n{e.stderr}")
            raise
        except Exception as e:
            print(f"Error running ORCA: {str(e)}")
            raise

        # Parse output
        self.results = self.parse_output()

        # Add configuration to results
        self.results["config"] = self.config

        # Clean up temporary files
        self.clean_up()
            
    def check_status(self):
        """
        Check the status of the ORCA output file.

        Returns:
            bool: True if the output file exists and contains the line "ORCA TERMINATED NORMALLY" in the last 5 lines, False otherwise.
        """
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
            return False
    
    def parse_output(self):
        """
        Parse ORCA output and property files.

        Returns:
            dict: A dictionary containing parsed results from the ORCA calculation.
        """
        if not self.check_status():
            raise RuntimeError("Calculation not complete or failed.")
            
        parser = OrcaOutput()
        
        # Parse property file if it exists
        if self.property_file.exists():
            with open(self.property_file, 'r') as f:
                results = parser.parse_orca_output(f.read())
        else:
            print("Warning: Property file not found.")
            results = None
            
        return results
    
    def clean_up(self):
        """
        Clean up calculation files.
        """
        for file in self.work_dir.iterdir():
            if not any(file.match(pattern) for pattern in self.PATTERNS_TO_KEEP):
                file.unlink()

    def get_molecule(self):
        """
        Return the last molecule from ORCA output as an ASE Atoms object.

        Returns:
            ase.Atoms: The last geometry of the calculation as an ASE Atoms object.

        Raises:
            ValueError: If no results are available.
            FileNotFoundError: If the geometry file is not found.
        """
        if not self.results:
            raise ValueError("No results available. Run calculation first.")
        
        # Construct path to last geometry
        mol_path = self.work_dir / f"{self.base_name}.xyz"
        if not mol_path.exists():
            raise FileNotFoundError(f"Geometry file not found: {mol_path}")

        # Read last geometry from file
        mol = read(mol_path, format="xyz")
        
        return mol


# Example usage
if __name__ == "__main__":
    
    # Example configuration
    config = {
        "base": "orca",
        "type": "opt",
        "method": "b3lyp",
        "basis": "6-31g",
        "nprocs": "2",
        "mem_per_proc": "3000",
        "keywords": "nopop smallprint",
    }

    # Read molecule from xyz file
    mol = read("./test/water.xyz", format="xyz")
    
    # Create ORCA manager
    orca = Orca(config, work_dir="./test/orca")

    # Prepare input and run calculation
    orca.prepare_input(molecule=mol)
    orca.run()
    
    # Print results
    print(json.dumps(orca.results, indent=2))