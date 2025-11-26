# =============================================================================
# Imports
# =============================================================================

# Standard
from setuptools import setup

# Local application
from sklr._build_utils import cythonize_extensions

import numpy as np

# =============================================================================
# Constants
# =============================================================================

# The module name is not package metadata, and
# cannot be declared in the configuration file
MOD_NAME = "sklr"


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    setup(ext_modules=cythonize_extensions(MOD_NAME), include_dirs=[np.get_include()])
