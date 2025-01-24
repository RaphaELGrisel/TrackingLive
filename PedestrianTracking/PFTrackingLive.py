import cv2
import numpy as np
from ObservationModel.ParticleFilter import PF

class PedestrianTracker:
    def __init__(self, args):
        self.args = args
        self.PF = PF(args)

        # Define variance for motion model
        self.var_c = np.array([20, 30, 10, 10, 40, 40, 0.1])
        self.var_m = np.array([20, 30, 10, 10, 40, 40, 0.1])

        # Define bounds for motion model
        self.bounds_c = np.array([25, 25, 30, 100, 60, 150, 0.05])
        self.bounds_m = np.array([25, 25, 30, 100, 60, 150, 0.05])

    def track_pedestrian(self):
        """
        Tracking pedestrians using Particle Filter. The user selects the initial ROI.
        """
        # Access webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot access webcam.")
            return

        print("Press 'q' to quit.")
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            return

        # Allow the user to select the ROI
        print("Please select the Region of Interest (ROI) and press ENTER or SPACE.")
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")

        # Initialize state vector for target based on selected ROI
        x, y, w, h = roi
        vx_initial = 0
        vy_initial = 0
        target = np.array([[x], [y], [vx_initial], [vy_initial], [w], [h], [1]])

        # Initialize particles
        particles_m, weights_m = self.PF.initparticles(frame.shape)
        particles_c, weights_c = self.PF.initparticles(frame.shape)

        print("Start tracking...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame.")
                break

            # Propagate particles
            particles_m = self.PF.motion.propagate(particles_m, self.args.dt, self.var_m, frame.shape, self.bounds_m)
            particles_c = self.PF.motion.propagate(particles_c, self.args.dt, self.var_c, frame.shape, self.bounds_c)

            # Observation model based on color histogram
            obsmodel_colors = self.PF.observationmodel(self.args.OM.upper(),2)
            t_colors = obsmodel_colors.initialize(target, frame)
            p_colors = obsmodel_colors.initialize(particles_c, frame)
            weights_c = obsmodel_colors.likelihood(t_colors, p_colors, self.args.mu_c, self.args.sigma_c)

            # Estimate weighted mean of all particles as the final estimate
            if np.sum(weights_c) > 0:
                weights_c /= np.sum(weights_c)
            est_c = np.dot(particles_c, weights_c).reshape(7, 1)

            # Update target state
            target = est_c

            # Draw particles and the estimated bounding box
            for p in range(particles_c.shape[1]):
                cv2.rectangle(
                    frame,
                    (int(particles_c[0, p]), int(particles_c[1, p])),
                    (int(particles_c[0, p] + particles_c[4, p]), int(particles_c[1, p] + particles_c[5, p])),
                    (0, 128, 255),
                    1,
                )

            # Draw the estimated bounding box
            cv2.rectangle(
                frame,
                (int(est_c[0]), int(est_c[1])),
                (int(est_c[0] + est_c[4]), int(est_c[1] + est_c[5])),
                (0, 255, 0),
                2,
            )

            # Show the frame
            cv2.imshow("Tracking", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Get input arguments from shell
    import argparse

    parser = argparse.ArgumentParser("Object Tracking with Particle Filter")

    # General configs for estimation
    parser.add_argument("--N", default=10, type=int, help="Specify number of particles")
    parser.add_argument("--alpha", default=0.1, type=float, help="Specify value fraction on how much we trust weighted estimate over initial target")
    parser.add_argument("--balance", default=True, type=bool, help="Specify whether or not balance filtering should be used")
    parser.add_argument("--thresh_c", default=0.01, type=float, help="Specify estimation threshold for weights of color-based state vector")
    parser.add_argument("--dt", default=0.5, type=float, help="Specify time step")
    parser.add_argument("--OM", default="CLR", type=str, help="Specify observation model to use. Choose CLR or CM")
    parser.add_argument("--mu_c", default=0.5, type=float, help="Specify mean of color-based observation model")
    parser.add_argument("--sigma_c", default=1.2, type=float, help="Specify variance of color-based observation model")

    parser.add_argument("--resampling", default="SYS", type=str,help="Specify which resampling method to use. Choose either VAN or SYS")

    args = parser.parse_args()

    # Run Tracker
    tracker = PedestrianTracker(args)
    tracker.track_pedestrian()
