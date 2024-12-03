import numpy as np
import os

# For gp_casadi
from src.utils.utils import safe_mkdir_recursive
from src.utils.DirectoryConfig import DirectoryConfig as DirConfig

def distance_maximizing_points_1d(points, n_train_points, dense_gp=None):
    """
    Heuristic function for sampling training points in 1D (one input feature and one output prediction dimensions)
    :param points: dataset points for the current cluster. Array of shape Nx1
    :param n_train_points: Integer. number of training points to sample.
    :param dense_gp: A GP object to sample the points from, or None of the points will be taken directly from the data.
    :return:
    """
    closest_points = np.zeros(n_train_points, dtype=int if dense_gp is None else float)

    if dense_gp is not None:
        n_train_points -= 1

    # Fit histogram in data with as many bins as the number of training points
    a, b = np.histogram(points, bins=n_train_points)
    hist_indices = np.digitize(points, b) - 1

    # Pick as training value the median or mean value of each bin
    for i in range(n_train_points):
        bin_values = points[np.where(hist_indices == i)]
        if len(bin_values) < 1:
            closest_points[i] = np.random.choice(np.arange(len(points)), 1)
            continue
        if divmod(len(bin_values), 2)[1] == 0:
            bin_values = bin_values[:-1]

        if dense_gp is None:
            # If no dense GP, sample median points in each bin from training set
            bin_median = np.median(bin_values)
            median_point_id = np.where(points == bin_median)[0]
            if len(median_point_id) > 1:
                closest_points[i] = median_point_id[0]
            else:
                closest_points[i] = median_point_id
        else:
            # If with GP, sample mean points in each bin from GP
            bin_mean = np.min(bin_values)
            closest_points[i] = bin_mean

    if dense_gp is not None:
        # Add dimension axis 0
        closest_points[-1] = np.max(points)
        closest_points = closest_points[np.newaxis, :]

    return closest_points

#########################################################
#                    For gp_casadi
#########################################################
def get_model_dir_and_file(model_options):
    directory = os.path.join(DirConfig.GP_MODELS_DIR, str(model_options["git"]), str(model_options["model_name"]))

    model_params = model_options["params"]
    file_name = ''
    model_vars = list(model_params.keys())
    model_vars.sort()
    for i, param in enumerate(model_vars):
        if i > 0:
            file_name += '__'
        file_name += 'no_' if not model_params[param] else ''
        file_name += param

    return directory, file_name

def safe_mknode_recursive(destiny_dir, node_name, overwrite):
    safe_mkdir_recursive(destiny_dir)
    if overwrite and os.path.exists(os.path.join(destiny_dir, node_name)):
        os.remove(os.path.join(destiny_dir, node_name))
    if not os.path.exists(os.path.join(destiny_dir, node_name)):
        os.mknod(os.path.join(destiny_dir, node_name))
        return False
    return True

def make_bz_matrix(x_dims, u_dims, x_feats, u_feats):
    """
    Generates the Bz matrix for the GP augmented MPC.
    :param x_dims: dimensionality of the state vector
    :param u_dims: dimensionality of the input vector
    :param x_feats: array with the indices of the state vector x used to make the first part of the GP feature vector z
    :param u_feats: array with the indices of the input vector u used to make the second part of the GP feature vector z
    :return:  The Bz matrix to map from input x and u features to the z feature vector.
    """

    bz = np.zeros((len(x_feats), x_dims))
    for i in range(len(x_feats)):
        bz[i, x_feats[i]] = 1
    bzu = np.zeros((len(u_feats), u_dims))
    for i in range(len(u_feats)):
        bzu[i, u_feats[i]] = 1
    bz = np.concatenate((bz, np.zeros((len(x_feats), u_dims))), axis=1)
    bzu = np.concatenate((np.zeros((len(u_feats), x_dims)), bzu), axis=1)
    bz = np.concatenate((bz, bzu), axis=0)
    return bz
