import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate    N = path.shape[0]
    t_nominal = np.zeros(N)
    for i in range(N-1):
        t_nominal[i+1] = t_nominal[i] +  np.linalg.norm(path[i+1,:]-path[i,:])/V_des
    t_smoothed = np.arange(0, t_nominal[-1], dt)

    tck_x = scipy.interpolate.splrep(t_nominal, path[:,0], k=3, s=alpha)
    tck_y = scipy.interpolate.splrep(t_nominal, path[:,1], k=3, s=alpha)
    x_d = scipy.interpolate.splev(t_smoothed, tck_x, der=0)
    y_d = scipy.interpolate.splev(t_smoothed, tck_y, der=0)
    xd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=1)
    yd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=1)
    theta_d = np.arctan2(yd_d, xd_d)
    xdd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=2)
    ydd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=2)

    """
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.

    path = np.array(path)
    N = path.shape[0]
    t_nominal = np.zeros(N)
    for i in range(N-1):
        t_nominal[i+1] = t_nominal[i] +  np.linalg.norm(path[i+1,:]-path[i,:])/V_des
    t_smoothed = np.arange(0, t_nominal[-1], dt)

    tck_x = scipy.interpolate.splrep(t_nominal, path[:,0], k=3, s=alpha)
    tck_y = scipy.interpolate.splrep(t_nominal, path[:,1], k=3, s=alpha)
    x_d = scipy.interpolate.splev(t_smoothed, tck_x, der=0)
    y_d = scipy.interpolate.splev(t_smoothed, tck_y, der=0)
    xd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=1)
    yd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=1)
    theta_d = np.arctan2(yd_d, xd_d)
    xdd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=2)
    ydd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=2)

    
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return traj_smoothed, t_smoothed



