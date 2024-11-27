""" Set of tunable parameters for the Simplified Simulator and model fitting.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os


class DirectoryConfig:
    """
    Class for storing directories within the package
    """

    _dir_path = os.path.dirname(os.path.realpath(__file__))
    CONFIG_DIR = os.path.join(_dir_path, '..', '..', 'config')
    FLIGHT_DATA_DIR = os.path.join(_dir_path, '..', '..', 'results')
    GP_MODELS_DIR = os.path.join(_dir_path, '..', '..', 'gp_models')
    ACADOS_MODEL_DIR = os.path.join(_dir_path, '..', '..', 'acados_ocp')
    QUAD_PARAM_DIR = os.path.join(_dir_path, '..', '..', 'quads')
