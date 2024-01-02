import numpy as np
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

WORLD_FILE = "data/world.dat"
SENSOR_DATA_FILE = "data/sensor_data.dat"

def plot_state(mu, sigma, landmarks, map_limits):
    lx = [landmarks[i + 1][0] for i in range(len(landmarks))]
    ly = [landmarks[i + 1][1] for i in range(len(landmarks))]

    estimated_pose = mu
    covariance = sigma[0:2, 0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:, max_ind]
    max_eigval = eigenvals[max_ind]

    min_ind = 0 if max_ind == 1 else 1
    min_eigvec = eigenvecs[:, min_ind]
    min_eigval = eigenvals[min_ind]

    chisquare_scale = 2.2789

    width = 2 * np.sqrt(chisquare_scale * max_eigval)
    height = 2 * np.sqrt(chisquare_scale * min_eigval)
    angle = np.arctan2(max_eigvec[1], max_eigvec[0])

    ell = Ellipse(xy=[estimated_pose[0], estimated_pose[1]], width=width, height=height, angle=angle / np.pi * 180)
    ell.set_alpha(0.25)

    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'ro', markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',
               scale_units='xy')
    plt.axis(map_limits)

    plt.grid(True)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Robot Localization with EKF')

    plt.pause(0.01)


def prediction_step(odometry, mu, sigma):
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    Q = np.array([[0.2, 0.0, 0.0],
                  [0.0, 0.2, 0.0],
                  [0.0, 0.0, 0.02]])

    x_new = x + delta_trans * np.cos(theta + delta_rot1)
    y_new = y + delta_trans * np.sin(theta + delta_rot1)
    theta_new = theta + delta_rot1 + delta_rot2

    G = np.array([
        [1, 0, -delta_trans * np.sin(theta + delta_rot1)],
        [0, 1, delta_trans * np.cos(theta + delta_rot1)],
        [0, 0, 1]
    ])

    mu = np.array([x_new, y_new, theta_new])
    sigma = G @ sigma @ G.T + Q

    return mu, sigma

def correction_step(sensor_data, mu, sigma, landmarks):
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    ids = sensor_data['id']
    ranges = sensor_data['range']

    H = []
    Z = []
    expected_ranges = []
    for i in range(len(ids)):
        lm_id = ids[i]
        meas_range = ranges[i]
        lx = landmarks[lm_id][0]
        ly = landmarks[lm_id][1]

        range_exp = np.sqrt((lx - x) ** 2 + (ly - y) ** 2)

        H_i = [(x - lx) / range_exp, (y - ly) / range_exp, 0]
        H.append(H_i)
        Z.append(meas_range)
        expected_ranges.append(range_exp)

    R = 0.5 * np.eye(len(ids))

    K_help = np.dot(sigma, np.transpose(H))
    K_denom = np.dot(np.dot(H, sigma), np.transpose(H)) + R
    K = np.dot(K_help, np.linalg.inv(K_denom))

    mu = mu + np.dot(K, (np.array(Z) - np.array(expected_ranges)))
    sigma = np.dot((np.eye(3) - np.dot(K, H)), sigma)

    return mu, sigma


def main():
    landmarks = read_world(WORLD_FILE)
    sensor_readings = read_sensor_data(SENSOR_DATA_FILE)

    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    map_limits = [-1, 12, -1, 10]

    for timestep in range(len(sensor_readings) // 2):
        plot_state(mu, sigma, landmarks, map_limits)
        mu, sigma = prediction_step(sensor_readings[timestep, 'odometry'], mu, sigma)

        if (timestep, 'sensor') in sensor_readings:
            mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

    plt.show(block=True)



if __name__ == "__main__":
    main()
