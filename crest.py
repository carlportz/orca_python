import subprocess
import json
from pathlib import Path
import shutil
import socket

from ase import Atoms
from ase.io import read, write

class CREST_input:
    """
    Class to generate CREST input files from a configuration dictionary.

    Attributes:
        VALID_TYPES (dict): Valid CREST calculation types.
        VALID_METHODS (dict): Valid CREST methods.
        config (dict): Configuration dictionary.

    Methods:
        __init__(config):
            Initialize the CREST_input object with a configuration dictionary.
        _validate_config():
            Validate the configuration dictionary.
        _generate_command():
            Generate the main CREST command line options based on the configuration.
        _generate_blocks():
            Generate the input blocks for geometry constraints.
        generate_input(work_dir, molecule=None):
            Generate the complete CREST input file and command line options.
        write_input(filename, work_dir, molecule=None):
            Write the CREST input and coordinates to a file, and extend the command line options.
    """
    
    # Valid CREST calculation types
    VALID_TYPES = {
        "conf": "",                   # Conformer search
        "nci": "--nci",               # Conformer search with non-covalent interactions
    }

    # Valid CREST methods
    VALID_METHODS = {
        "gfn1": "--gfn 1",
        "gfn2": "--gfn 2",
        "gfnff": "--gfnff",
    }
    
    def __init__(self, config):
        """
        Initialize the CREST_input object with a configuration dictionary.

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
                raise ValueError(f"Missing required configuration key: {key}.")

        # Validate calculation type
        if self.config["type"].lower() not in self.VALID_TYPES:
            raise ValueError(f"Invalid calculation type: {self.config['type']}. Valid types are: {', '.join(self.VALID_TYPES.keys())}")

        # Validate method if specified
        if "method" in self.config and self.config["method"].lower() not in self.VALID_METHODS:
            raise ValueError(f"Invalid method: {self.config['method']}. Valid methods are: {', '.join(self.VALID_METHODS.keys())}")

    def _generate_command(self):
        """
        Generate the main CREST command line options.

        Returns:
            str: The generated CREST command line options as a single string.
        """
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

        # Add other keywords if specified
        if "keywords" in self.config:
            parts.append(self.config["keywords"])
                
        return " ".join(parts)

    def _generate_blocks(self):
        """
        Generate the input blocks for geometry constraints.

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
        
        return "\n".join(filter(bool, blocks))  # filter out empty strings

    def generate_input(self):
        """
        Generate the complete CREST input file and command line options.

        Returns:
            tuple: A tuple containing the command line options and input content.
        """
        # Generate the main command line options
        command = self._generate_command()

        # Generate the input blocks, if necessary
        if "constraints" in self.config:
            input_content = self._generate_blocks()
        else:
            input_content = None

        return command, input_content

    def write_input(self, filename, work_dir, molecule=None):
        """
        Write the CREST input and coordinates to a file, and extend the command line options.

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
            
            command += f" --cinp {filename}"

        # Write the coordinates to a separate file
        coords_file = work_dir / "crest_input.xyz"
        write(coords_file, molecule, format='xyz')

        # Add coordinates file to the command
        command = f"{coords_file}" + command

        return command


class CREST:
    """
    Class to manage CREST calculations: input generation, execution, and output parsing.

    Attributes:
        config (dict): Dictionary containing calculation parameters.
        work_dir (Path): Working directory for the calculation.
        crest_cmd (str): Path to CREST executable.
        input_file (Path): Path to the CREST input file.
        output_file (Path): Path to the CREST output file.
        results (dict): Parsed results from the CREST calculation.

    Constants:
        PATTERNS_TO_KEEP (list): List of file patterns to keep after cleaning.

    Methods:
        prepare_input(molecule=None):
            Generate CREST input file, if necessary.
        run():
            Execute CREST calculation and wait for it to complete.
        check_status():
            Check if the calculation has completed and was successful.
        parse_output():
            Parse CREST energy file.
        clean_up(keep_main_files=True):
            Clean up calculation files.
        get_ensemble():
            Return optimized ensemble from CREST output as ASE Atoms object.
    """

    # Constant for file patterns to keep
    PATTERNS_TO_KEEP = ["*.inp", 
                        "*.out", 
                        "*.xyz", 
                        "*.coord"]
    
    def __init__(self, config, crest_cmd=None, work_dir=None):
        """
        Initialize the CREST class with configuration, command path, and working directory.

        Args:
            config (dict): Configuration dictionary containing necessary parameters.
            crest_cmd (str, optional): Path to the CREST executable. Defaults to the system path.
            work_dir (str or Path, optional): Path to the working directory. Defaults to the current working directory.
        """
        self.config = config
        self.work_dir = Path.cwd().resolve() if work_dir is None else Path(work_dir).resolve()
        self.crest_cmd = shutil.which("crest") if crest_cmd is None else crest_cmd
        
        # Create working directory if it doesn't exist
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file paths
        self.input_file = None
        self.output_file = None
        self.results = None
        
    def prepare_input(self, molecule=None):
        """
        Generate CREST input file, if necessary.

        Args:
            molecule (ase.Atoms, optional): The molecular structure to be used in the CREST calculation. If not provided, a default structure will be used.
        """
        self.input_file = self.work_dir / "crest.inp"
        self.output_file = self.work_dir / "crest.out"
        
        # Generate input file if necessary
        generator = CREST_input(self.config)
        self.cmd_options = generator.write_input(self.input_file, self.work_dir, molecule=molecule)
        
    def run(self):
        """
        Execute CREST calculation and wait for it to complete.

        Raises:
            ValueError: If the input file is not prepared.
            subprocess.CalledProcessError: If the CREST calculation fails.
            Exception: If there is an error running CREST.
        """
        if not self.input_file:
            raise ValueError("Input file not prepared. Call prepare_input() first.")

        print(f"Crest running in {self.work_dir} on {socket.gethostname()}")
            
        # Clean up temporary files
        self.clean_up()

        # Prepare command
        cmd = f"{self.crest_cmd} {self.cmd_options} > {self.output_file}"
        
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
        self.results["config"] = self.config

        # Clean up temporary files
        self.clean_up()
            
    def check_status(self):
        """
        Check if the calculation has completed and was successful.

        Returns:
            bool: True if the output file exists and contains the line "CREST terminated normally." in the last 5 lines, False otherwise.
        """
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
        """
        Parse CREST energy file.

        Returns:
            dict: A dictionary containing parsed energies from the CREST calculation.
        """
        if not self.check_status():
            raise RuntimeError("Calculation not complete or failed.")
            
        # Parse energy file if it exists
        energy_file = self.work_dir / "crest.energies"
        if energy_file.exists():
            with open(energy_file, 'r') as f:
                energies = [float(line.split()[1]) for line in f if line.strip()]
        else:
            print("Warning: Energy file not found.")
            energies = None
            
        return {"energies": energies}
    
    def clean_up(self):
        """
        Clean up calculation files.
        """
        for file in self.work_dir.iterdir():
            if not any(file.match(pattern) for pattern in self.PATTERNS_TO_KEEP):
                file.unlink()

    def get_ensemble(self):
        """
        Return optimized ensemble from CREST output as a list of ASE Atoms objects.

        Returns:
            list[ase.Atoms]: The optimized ensemble of conformers as a list of ASE Atoms objects.

        Raises:
            ValueError: If no results are available.
            FileNotFoundError: If the ensemble file is not found.
        """
        if not self.results:
            raise ValueError("No results available. Run calculation first.")
        
        # Construct path to last geometry
        mol_path = self.work_dir / "crest_conformers.xyz"
        if not mol_path.exists():
            raise FileNotFoundError(f"Ensemble file not found: {mol_path}")

        # Read all geometries from file
        ensemble = read(mol_path, format="xyz", index=":")
        
        return ensemble


# Example usage
if __name__ == "__main__":

    # Check if the current hostname is not wuxcs
    assert socket.gethostname() != "wuxcs", "This script should not be run on the wuxcs."
    
    # Example configuration
    config = {
        "type": "conf",
        "method": "gfn2",
        "nprocs": "20",
        #"constraints": "dihedral: 1,2,3,4,90.0",
        #"constraints": "angle: 2,3,4,180.0",
        #"force constant": "0.25",
        #"solvent": "water",
    }

    # Read molecule from xyz file
    mol = read("./test/n-butane.xyz", format='xyz')

    # Create CREST manager
    crest = CREST(config, work_dir="/scratch/2329184/")

    # Prepare input and run calculation
    crest.prepare_input(molecule=mol)
    crest.run()
    
    # Parse results (raw data)
    print(json.dumps(crest.results, indent=2))

