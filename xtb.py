import subprocess
import json
from pathlib import Path
import shutil
import socket
import os

from ase import Atoms
from ase.io import read, write

class XtbInput:
    """
    Class to generate XTB input files from a configuration dictionary.

    Attributes:
        VALID_TYPES (dict): Valid XTB calculation types.
        VALID_METHODS (dict): Valid XTB methods.
        config (dict): Configuration dictionary.

    Methods:
        __init__(config):
            Initialize the XTB_input object with a configuration dictionary.
        _validate_config():
            Validate the configuration dictionary.
        _generate_command():
            Generate the main XTB command line options based on the configuration.
        _generate_blocks():
            Generate the input blocks for geometry constraints and relaxed scans.
        generate_input(work_dir, molecule=None):
            Generate the complete XTB input file and command line options.
        write_input(filename, work_dir, molecule=None):
            Write the XTB input and coordinates to a file, and extend the command line options.
    """
    
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
        """
        Initialize the XTB_input object with a configuration dictionary.

        Args:
            config (dict): A dictionary containing configuration parameters.
        """
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """
        Validate the configuration dictionary.

        Raises:
            ValueError: If any required key is missing or contains an invalid value.
        """
        required_keys = ["type"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate calculation type
        if self.config["type"].lower() not in self.VALID_TYPES:
            raise ValueError(f"Invalid calculation type: {self.config['type']}. Valid types are: {', '.join(self.VALID_TYPES.keys())}")

        # Validate method if specified
        if "method" in self.config and self.config["method"].lower() not in self.VALID_METHODS:
            raise ValueError(f"Invalid method: {self.config['method']}. Valid methods are: {', '.join(self.VALID_METHODS.keys())}")

    def _generate_command(self):
        """
        Generate the main XTB command line options.

        Returns:
            str: The generated XTB command line options as a single string.
        """
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

        # Add other keywords if specified
        if "keywords" in self.config:
            parts.append(self.config["keywords"])
                
        return " ".join(parts)

    def _generate_blocks(self):
        """
        Generate the input blocks for geometry constraints and relaxed scans.

        Returns:
            str: A string containing the concatenated blocks of settings.
        """
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

    def generate_input(self):
        """
        Generate the complete XTB input file and command line options.

        Returns:
            tuple: A tuple containing the command line options and input content.
        """
        # Generate the main command line options
        command = self._generate_command()

        # Generate the input blocks, if necessary
        if "constraints" in self.config or "scan" in self.config:
            input_content = self._generate_blocks()
        else:
            input_content = None

        return command, input_content

    def write_input(self, filename, work_dir, molecule=None):
        """
        Write the XTB input and coordinates to a file, and extend the command line options.

        Args:
            filename (str): The name of the file to write the input to.
            work_dir (Path): The working directory for the calculation.
            molecule (ase.Atoms, optional): An ASE Atoms object representing the molecule.

        Returns:
            str: The complete command line options including the input and coordinates files.
        """
        if not molecule:
            molecule = Atoms("H", positions=[[0, 0, 0]])
            print("Warning: No molecule provided. Using default hydrogen atom.")

        # Generate input content and command
        command, input_content = self.generate_input()

        # Write the input file
        if input_content:
            with open(filename, 'w') as f:
                f.write(input_content)
            
            command += f" --input {filename}"

        # Write the coordinates to a separate file
        coords_file = os.path.join(work_dir, f"{self.config['base']}_input.xyz")
        write(coords_file, molecule, format='xyz')

        # Add coordinates file to the command
        command += f" -- {coords_file}"

        return command


class XtbOutput:
    """
    Class to handle XTB output file parsing.

    Methods:
        parse_xtb_output(content: str) -> dict:
            Parse XTB output file SUMMARY into a structured dictionary.
    """

    def parse_xtb_output(self, content):
        """
        Parse XTB output file SUMMARY sections into a structured dictionary.

        Args:
            content (str): The content of the XTB output file as a string.

        Returns:
            dict: A dictionary containing parsed data from the XTB output file.
        """
        result = {"Properties": []}
        lines = content.split('\n')
        summary_found = False
        dipole_found = False
        
        # Process lines
        for line in lines:
            line = line.strip()
            
            # Start of property section
            if 'SUMMARY' in line:
                summary = {}
                summary_found = True
                continue
            
            # End of property section
            if '..........' in line and summary_found:
                result["Properties"].append(summary)
                summary_found = False
            
            # Parse properties
            if summary_found:
                # Remove leading/trailing ':' and split by whitespace
                clean_line = line.replace(':', '').strip()
                parts = clean_line.split()
                
                if len(parts) >= 2:
                    # Join all parts except the last two (value and unit) as the key
                    key = ' '.join(parts[:-2]).strip()
                    value = float(parts[-2])  # Convert value to float
                    summary[key] = value 

            # Start of dipole section
            if 'molecular dipole' in line:
                dipole_found = True
                continue

            # End of dipole section
            if 'molecular quadrupole' in line and dipole_found:
                result["Properties"][-1]["Dipole"] = dipole
                dipole_found = False

            # Parse dipole
            if dipole_found and 'full:' in line:
                parts = line.strip().split()
                dipole = [float(x) for x in parts[1:4]]
        
        return result


class Xtb:
    """
    Class to manage XTB calculations: input generation, execution, and output parsing.

    Attributes:
        config (dict): Dictionary containing calculation parameters.
        work_dir (Path): Working directory for the calculation.
        xtb_cmd (str): Path to XTB executable.
        base_name (str): Base name for input, output, and property files.
        input_file (Path): Path to the XTB input file.
        output_file (Path): Path to the XTB output file.
        results (dict): Parsed results from the XTB calculation.

    Constants:
        PATTERNS_TO_KEEP (list): List of file patterns to keep after cleaning.
        
    Methods:
        prepare_input(molecule=None):
            Generate XTB input file, if necessary.
        run():
            Execute XTB calculation and wait for it to complete.
        check_status():
            Check if the calculation has completed and was successful.
        parse_output():
            Parse XTB output file.
        clean_up(keep_main_files=True):
            Clean up calculation files.
        get_molecule():
            Return the last molecule from XTB output as an ASE Atoms object.
    """

    # Constant for file patterns to keep
    PATTERNS_TO_KEEP = ["*.inp", 
                        "*.out", 
                        "*.xyz", 
                        "*.coord", 
                        "*.log"]
    
    def __init__(self, config, xtb_cmd=None, work_dir=None):
        """
        Initialize the XTB class with configuration, command path, and working directory.

        Args:
            config (dict): Configuration dictionary containing necessary parameters.
            xtb_cmd (str, optional): Path to the XTB executable. Defaults to the system path.
            work_dir (str or Path, optional): Path to the working directory. Defaults to the current working directory.
        """
        self.config = config
        self.work_dir = Path.cwd().resolve() if work_dir is None else Path(work_dir).resolve()
        self.xtb_cmd = shutil.which("xtb") if xtb_cmd is None else xtb_cmd
        
        # Create working directory if it doesn't exist
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file paths using base name from config
        self.base_name = self.config['base'] if 'base' in self.config else "orca"
        self.input_file = None
        self.output_file = None
        self.results = None
        
    def prepare_input(self, molecule=None):
        """
        Generate XTB input file, if necessary.

        Args:
            molecule (ase.Atoms, optional): The molecular structure to be used in the XTB calculation. If not provided, a default structure will be used.
        """
        self.input_file = self.work_dir / f"{self.base_name}.inp"
        self.output_file = self.work_dir / f"{self.base_name}.out"
        
        # Generate input file if necessary
        generator = XtbInput(self.config)
        self.cmd_options = generator.write_input(self.input_file, self.work_dir, molecule=molecule)
        
    def run(self):
        """
        Execute XTB calculation and wait for it to complete.

        Raises:
            ValueError: If the input file is not prepared.
            subprocess.CalledProcessError: If the XTB calculation fails.
            Exception: If there is an error running XTB.
        """
        if not self.input_file:
            raise ValueError("Input file not prepared. Call prepare_input() first.")

        if "verbose" in self.config and self.config["verbose"]:
            print(f"xtb running in {self.work_dir} on {socket.gethostname()}")
        
        # Clean up temporary files
        self.clean_up()

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
        self.results["config"] = self.config

        # Clean up temporary files
        self.clean_up()

    def check_status(self):
        """
        Check if the calculation has completed and was successful.

        Returns:
            bool: True if the output file exists, False otherwise.
        """
        if not self.output_file.exists():
            return False
        else:
            return True
            
    def parse_output(self):
        """
        Parse XTB output file.

        Returns:
            dict: A dictionary containing parsed results from the XTB calculation.
        """
        if not self.check_status():
            raise RuntimeError("Calculation not complete or failed.")
            
        parser = XtbOutput()
        
        # Parse output file if it exists
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                self.results = parser.parse_xtb_output(f.read())
        else:
            print("Warning: Output file not found.")
            self.results = None
            
        return self.results
    
    def clean_up(self):
        """
        Clean up calculation files.
        """
        for file in self.work_dir.iterdir():
            if not any(file.match(pattern) for pattern in self.PATTERNS_TO_KEEP):
                file.unlink()

    def get_molecule(self):
        """
        Return the last molecule from XTB output as an ASE Atoms object.

        Returns:
            ase.Atoms: The last geometry of the calculation as an ASE Atoms object.

        Raises:
            ValueError: If no results are available.
            FileNotFoundError: If the geometry file is not found.
        """
        if not self.results:
            raise ValueError("No results available. Run calculation first.")
        
        # Construct path to last geometry
        mol_path = os.path.join(self.work_dir, f"{self.base_name}.xtbopt.xyz")
        if not os.path.exists(mol_path):
            raise FileNotFoundError(f"Geometry file not found: {mol_path}")

        # Read last geometry from file
        mol = read(mol_path, format="xyz")
        
        return mol


# Example usage
if __name__ == "__main__":
    
    # Example configuration
    config = {
        "base": "xtb",
        "type": "opt",
        "method": "gfn2",
        "nprocs": "2",
        "constraints": "angle: 2, 1, 3, 120.0",
        "force constant": 1.5,
    }
 
    # Read molecule from xyz file
    mol = read("./test/water.xyz", format='xyz')
    
    # Create XTB manager
    xtb = Xtb(config, work_dir="./test/xtb")

    # Prepare input and run calculation
    xtb.prepare_input(molecule=mol)
    xtb.run()
    
    # Print results
    print(json.dumps(xtb.results, indent=2))
