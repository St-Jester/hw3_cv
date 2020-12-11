# Develop a python application that uses Kalman filter for smoothing the output of the mouse
# pointer. Define process model, transition matrix and system noise. Write code as python
# application with comments.

import cv2 as cv
import numpy as np
from math import cos, sin, sqrt


class KalmanFilter:

    def __init__(self, state_est, covar_estimate, state_transition_model, processNoiseCov, state_measurment,
                 observation_model, observation_NoiseCov):
        """
        Initialise the filter
        Args:
            state_est: State estimate
            covar_estimate: Estimate covariance
            state_transition_model: State transition model
            processNoiseCov: Process noise covariance
            state_measurment: Measurement of the state X
            observation_model: Observation model
            observation_NoiseCov: Observation noise covariance


             X: State estimate
            P: Estimate covariance
            F: State transition model
            Q: Process noise covariance
            Z: Measurement of the state X
            H: Observation model
            R: Observation noise covariance
        """
        self.state_est = state_est
        self.covar_estimate = covar_estimate
        self.state_transition_model = state_transition_model
        self.processNoiseCov = processNoiseCov
        self.state_measurment = state_measurment
        self.observation_model = observation_model
        self.observation_NoiseCov = observation_NoiseCov

    def predict(self, noise):
        """
        Predict the future state

        Returns:
            updated (state_est_X, covar_estimate_P)
        """
        # Project the state ahead
        self.state_est = self.state_transition_model @ self.state_est
        self.state_est += noise.T
        self.covar_estimate = self.state_transition_model @ self.covar_estimate @ (
            self.state_transition_model.T) + self.processNoiseCov

        return (self.state_est, self.covar_estimate)

    def correct(self, state_measurment):
        """
        Update the Kalman Filter from a measurement
        Args:
            Z: State measurement
        Returns:
            updated (self.state_est, self.covar_estimate)
            :param state_measurment:
        """
        K = self.covar_estimate @ (self.observation_model.T) @ np.linalg.inv(
            self.observation_model @ self.covar_estimate @ (self.observation_model.T) + self.observation_NoiseCov)

        self.state_est += K @ (state_measurment - self.observation_model @ self.state_est)
        self.covar_estimate = (np.identity(
            self.covar_estimate.shape[1]) - K @ self.observation_model) @ self.covar_estimate
        return self.state_est


# press ESC to quit

ix, iy = -1, -1
img_height = 512
img_width = 512
img = np.zeros((img_height, img_width, 3), np.uint8)


def follow_cursor(event, x, y, flags, param):
    global img
    if event == cv.EVENT_MOUSEMOVE:
        # red = movement of the cursor
        # cv.circle(img, (x, y), 1, (0, 0, 255), -1)

        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
        # noise
        w = np.array([np.random.multivariate_normal([0, 0, 0, 0], processNoiseCov)])

        (current_prediction, current_covar_noise_est) = kalman.predict(w)

        cmx, cmy = current_measurement[0], current_measurement[1]
        cpx, cpy = current_prediction[0], current_prediction[1]

        img = np.zeros((img_height, img_width, 3), np.uint8)

        cv.putText(img, "Real position: ({:.1f}, {:.1f})".format(np.float(cmx), np.float(cmy)),
                   (30, 50), cv.FONT_HERSHEY_DUPLEX, 0.8, (50, 150, 0))
        cv.putText(img, "Predicted position: ({:.1f}, {:.1f})".format(np.float(cpx), np.float(cpy)),
                   (30, 100), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255))
        # cv.putText(img, "noise_x position: ({:.1f}, {:.1f})".format(np.float(noise_x), np.float(noise_y)),
        #            (30, 150), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255))
        cv.circle(img, (cmx, cmy), 4, (50, 150, 0), -1)  # current measured point
        cv.circle(img, (cpx, cpy), 4, (0, 0, 255), -1)  # current predicted point

        #
        # for inx in w.T:
        #     print(inx[0])
        #     cv.circle(img, (cmx + inx[0], cmy + inx[0]), 1, (0, 255, 255), -1)  # current predicted point
        #     cv.circle(img, (cmx + inx[0], cmy), 1, (0, 255, 255), -1)  # current predicted point
        #     cv.circle(img, (cmx, cmy + inx[0]), 1, (0, 255, 255), -1)  # current predicted point
        #     cv.circle(img, (cmx - inx[0], cmy - inx[0]), 1, (0, 255, 255), -1)  # current predicted point
        #     cv.circle(img, (cmx - inx[0], cmy), 1, (0, 255, 255), -1)  # current predicted point
        #     cv.circle(img, (cmx, cmy - inx[0]), 1, (0, 255, 255), -1)  # current predicted point
        #     cv.circle(img, (cmx + inx[0], cmy - inx[0]), 1, (0, 255, 255), -1)  # current predicted point
        #     cv.circle(img, (cmx + inx[0], cmy), 1, (0, 255, 255), -1)  # current predicted point
        #     cv.circle(img, (cmx, cmy - inx[0]), 1, (0, 255, 255), -1)  # current predicted point
        #
        #     cv.circle(img, (cmx - inx[0], cmy + inx[0]), 1, (0, 255, 255), -1)  # current predicted point
        #     cv.circle(img, (cmx, cmy + inx[0]), 1, (0, 255, 255), -1)  # current predicted point
        #     cv.circle(img, (cmx - inx[0], cmy), 1, (0, 255, 255), -1)  # current predicted point

        # cv.circle(img, (cmx + inx[1], cmy + inx[1]), 2, (0, 255, 255), -1)  # current predicted point
        # cv.circle(img, (cmx + inx[2], cmy + inx[2]), 2, (0, 255, 255), -1)  # current predicted point
        # cv.circle(img, (cmx + inx[3], cmy + inx[3]), 2, (0, 255, 255), -1)  # current predicted point

        kalman.correct(current_measurement)

    return


cv.namedWindow('image')
cv.setMouseCallback('image', follow_cursor)

stateMatrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
estimateCovariance = np.eye(stateMatrix.shape[0])
transitionMatrix = np.array([[1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], np.float32)

processNoiseCov = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], np.float32) * 0.05

measurementStateMatrix = np.zeros((2, 1), np.float32)

observationMatrix = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]], np.float32)

measurementNoiseCov = np.array([[1, 0],
                                [0, 1]], np.float32) * 1

kalman = KalmanFilter(state_est=stateMatrix,
                      covar_estimate=estimateCovariance,
                      state_transition_model=transitionMatrix,
                      processNoiseCov=processNoiseCov,
                      state_measurment=measurementStateMatrix,
                      observation_model=observationMatrix,
                      observation_NoiseCov=measurementNoiseCov)

while True:
    cv.imshow('image', img)
    k = cv.waitKey(10) & 0xFF

    if k == 27:
        break

cv.destroyAllWindows()
#
# #
# kalman = cv.KalmanFilter(4, 2, 1)
# #
# # cv.namedWindow('image')
# # cv.setMouseCallback('image', follow_cursor)
# # cv.waitKey()
# #
#
#
# while 1:
#     kalman.transitionMatrix = np.array([[1, 0, 1, 0],
#                                         [0, 1, 0, 1],
#                                         [0, 0, 1, 0],
#                                         [0, 0, 0, 1]], np.float32)
#
#     kalman.measurementMatrix = np.array([[1, 0, 0, 0],
#                                          [0, 1, 0, 0]], np.float32)
#
#     cv.setIdentity(kalman.processNoiseCov, 1e-5)
#     cv.setIdentity(kalman.measurementNoiseCov, 1e-1)
#     cv.setIdentity(kalman.errorCovPost, 1)
#     kalman.statePost = 0.1 * np.random.randn(2, 1)
#     state = np.zeros(4, np.float32)
#     cv.randn(state, 0, 0.1)
#
#     cv.imshow('image', img)
#     while True:
#         # def calc_point(angle):
#         #     return (np.around(img_width / 2 + img_width / 3 * cos(angle), 0).astype(int),
#         #             np.around(img_height / 2 - img_width / 3 * sin(angle), 1).astype(int))
#         #
#         #
#         # state_angle = state[0, 0]
#         # state_pt = calc_point(state_angle)
#         #
#         # prediction = kalman.predict()
#         # predict_angle = prediction[0, 0]
#         # predict_pt = calc_point(predict_angle)
#         #
#         # measurement = kalman.measurementNoiseCov * np.random.randn(1, 1)
#         #
#         # # generate measurement
#         # measurement = np.dot(kalman.measurementMatrix, state) + measurement
#         #
#         # measurement_angle = measurement[0, 0]
#         # measurement_pt = calc_point(measurement_angle)
#         #
#         # # plot points
#         # def draw_cross(center, color, d):
#         #
#         #     cv.line(img,
#         #             (center[0] - d, center[1] - d), (center[0] + d, center[1] + d),
#         #             color, 1, cv.LINE_AA, 0)
#         #     cv.line(img,
#         #             (center[0] + d, center[1] - d), (center[0] - d, center[1] + d),
#         #             color, 1, cv.LINE_AA, 0)
#         #
#         #
#         # img = np.zeros((img_height, img_width, 3), np.uint8)
#         # draw_cross(np.int32(state_pt), (255, 255, 255), 3)
#         # draw_cross(np.int32(measurement_pt), (0, 0, 255), 3)
#         # draw_cross(np.int32(predict_pt), (0, 255, 0), 3)
#         #
#         # cv.line(img, state_pt, measurement_pt, (0, 0, 255), 3, cv.LINE_AA, 0)
#         # cv.line(img, state_pt, predict_pt, (0, 255, 255), 3, cv.LINE_AA, 0)
#         #
#         # kalman.correct(measurement)
#         #
#         # process_noise = sqrt(kalman.processNoiseCov[0, 0]) * np.random.randn(2, 1)
#         # state = np.dot(kalman.transitionMatrix, state) + process_noise
#         measurement = np.zeros(2, np.float32)
#         statePt = (kalman.statePost[0], kalman.statePost[1])
#         prediction = kalman.predict()
#         predictPt = (prediction[0], prediction[1])
#         measurement[0] = statePt[0]
#         measurement[1] = statePt[1]
#         kalman.correct(measurement)
#         cv.circle((0, 255, 0), (ix, iy), 5, (0, 255, 0), 1)
#         cv.circle((255, 0, 0), (predictPt[0], predictPt[1]), 5, (0, 0, 255), 1)
#
#         cv.imshow("Kalman", img)
#
#         code = cv.waitKey(100)
#         if code != -1:
#             break
#     if cv.waitKey(20) & 0xFF == 27:
#         break
# cv.destroyAllWindows()

# transition_Matrix = [[1, 0, 1, 0],
#                      [0, 1, 0, 1],
#                      [0, 0, 1, 0],
#                      [0, 0, 0, 1]]
#
# observation_Matrix = [[1, 0, 0, 0],
#                       [0, 1, 0, 0]]


# while 1:
#     cv.imshow('image', img)
#     k = cv.waitKey(10) & 0xFF
#
#     if k == 27:
#         break
# cv.destroyAllWindows()
