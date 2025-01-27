import cv2
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim  # Updated import


def make_box(img, center, w, h):
    """Utility function to calculate box corners given a center, width, and height."""
    w_half = w // 2
    h_half = h // 2
    x, y = center

    pt1 = (int(x - w_half), int(y - h_half))
    pt2 = (int(x + w_half), int(y - h_half))
    pt3 = (int(x + w_half), int(y + h_half))
    pt4 = (int(x - w_half), int(y + h_half))

    cv2.line(img, pt1, pt2, [0, 0, 255], 2)
    cv2.line(img, pt2, pt3, [0, 0, 255], 2)
    cv2.line(img, pt3, pt4, [0, 0, 255], 2)
    cv2.line(img, pt4, pt1, [0, 0, 255], 2)

    return img, pt1, pt2, pt3, pt4


def get_ref_image(cam, similarity_measure, counter=10, size=100):
    """Extracts a reference image to be used as a comparison against different patches."""
    cd = counter * 3
    for i in range(counter * 3)[::-1]:
        time.sleep(0.1)
        ret, img = cam.read()
        if not ret:
            # Frame grab failed
            ref_img, ref_x, ref_y = None, None, None
            break

        h, w, d = img.shape
        ref_x, ref_y = w // 2, h // 2
        img = cv2.flip(img, 1)

        cv2.putText(img, str('Particle filter tracker test, similarity measure: ' + similarity_measure),
                    (5, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, [0, 0, 255])
        cv2.putText(img, str('Press "c" to capture image in box or wait for countdown: ' + str(int(cd / 3))),
                    (w // 10, h * 3 // 4), cv2.FONT_HERSHEY_PLAIN, 1.0, [0, 0, 255])

        if i % 3 == 0:
            cv2.imshow('Camera view', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
        else:
            img, pt1, pt2, pt3, pt4 = make_box(img, (w / 2, h / 2), size * w / w, size * w / h)
            ref_img = img[pt1[1]:pt4[1], pt1[0]:pt2[0]]
            cv2.imshow('Camera view', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break

        if i % 3 == 1:
            # Replace or remove beeper function
            pass  # Placeholder for beeper functionality
            cd -= 3

    # Replace or remove beeper function
    return ref_img, (ref_x, ref_y)


def calc_similarity(ref_img, patch, sigma, sim_type='MSE_grayscale'):
    """Calculates the similarity between the reference and the candidate patches."""
    # Prep grayscale
    color_ref_img = ref_img
    color_patch = patch
    gray_ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Select similarity measure
    if sim_type == 'MSE_color':
        mse = np.mean((color_ref_img - color_patch) ** 2) / color_ref_img.size

    elif sim_type == 'MSE_grayscale':
        mse = np.mean((gray_ref_img - gray_patch) ** 2) / color_ref_img.size

    elif sim_type == 'Covariance_color':
        # Cross correlation comparison on color channels
        b_ref, g_ref, r_ref = np.rollaxis(color_ref_img, axis=2)
        b_patch, g_patch, r_patch = np.rollaxis(color_patch, axis=2)

        b_ref_mean = np.mean(b_ref)
        g_ref_mean = np.mean(g_ref)
        r_ref_mean = np.mean(r_ref)

        b_patch_mean = np.mean(b_patch)
        g_patch_mean = np.mean(g_patch)
        r_patch_mean = np.mean(r_patch)

        sim_blue = np.sum((b_ref - b_ref_mean) * (b_patch - b_patch_mean)) / (np.std(b_ref) * np.std(b_patch))
        sim_green = np.sum((g_ref - g_ref_mean) * (g_patch - g_patch_mean)) / (np.std(g_ref) * np.std(g_patch))
        sim_red = np.sum((r_ref - r_ref_mean) * (r_patch - r_patch_mean)) / (np.std(r_ref) * np.std(r_patch))

        sim = (sim_blue + sim_green + sim_red) / 3.0

    elif sim_type == 'Covariance_grayscale':
        # Cross correlation comparison on grayscale
        gray_ref_img_mean = np.mean(gray_ref_img)
        gray_patch_mean = np.mean(gray_patch)

        sim = np.sum((gray_ref_img - gray_ref_img_mean) * (gray_patch - gray_patch_mean)) / (
                    np.std(gray_ref_img) * np.std(gray_patch))

    elif sim_type == 'SSIM_grayscale':
        # Comparison using scikit's SSIM
        sim = ssim(gray_patch, gray_ref_img)

    elif sim_type == 'MSE_histogram_grayscale':
        gray_hist_ref = cv2.calcHist([gray_ref_img], [0], None, [256], [0, 256])
        gray_hist_patch = cv2.calcHist([gray_patch], [0], None, [256], [0, 256])

        mse = np.mean((gray_hist_patch - gray_hist_ref) ** 2) / gray_ref_img.size

    elif sim_type == 'MSE_histogram_color':
        b_hist_ref = cv2.calcHist([color_ref_img], [0], None, [256], [0, 256])
        g_hist_ref = cv2.calcHist([color_ref_img], [1], None, [256], [0, 256])
        r_hist_ref = cv2.calcHist([color_ref_img], [2], None, [256], [0, 256])

        b_hist_patch = cv2.calcHist([color_patch], [0], None, [256], [0, 256])
        g_hist_patch = cv2.calcHist([color_patch], [1], None, [256], [0, 256])
        r_hist_patch = cv2.calcHist([color_patch], [2], None, [256], [0, 256])

        color_hist_ref = np.vstack((b_hist_ref, g_hist_ref, r_hist_ref))
        color_hist_patch = np.vstack((b_hist_patch, g_hist_patch, r_hist_patch))

        mse = np.mean((color_hist_patch - color_hist_ref) ** 2) / color_hist_ref.size

    else:
        sim = 0.0

    # Change the MSE value to a probability based on a Gaussian distribution
    if sim_type in ['MSE_color', 'MSE_grayscale', 'MSE_histogram_color', 'MSE_histogram_grayscale']:
        # Normal distribution probability for MSE
        sim = 1.0 / (2.0 * np.pi * sigma) * np.exp(-mse / (2.0 * sigma ** 2))

    # Set ranking type for the particle filter sort later
    if sim_type == 'something new and lowest value is best':
        ranking_type = 'ascending'
    else:
        ranking_type = 'descending'

    return sim, ranking_type


def resample_particles(particles, p_weights):
    """Extracts a new set of particles based on probability weights of prior particles."""
    num_particles = len(p_weights)
    idx = np.random.choice(range(num_particles), size=num_particles, p=p_weights, replace=True)
    new_particles = particles[idx]

    return new_particles


def get_patch(frame, w, h, x, y):
    """Extracts a new image patch from a frame based on given coordinates and patch dimensions."""
    # Adjust edges if beyond frame
    x = x + int(w // 2 - x) if int(x - w // 2) < 0 else x
    x = x - (int(x + w // 2) - len(frame[0])) if int(x + w // 2) > len(frame[0]) else x
    y = y + int(h // 2 - y) if int(y - h // 2) < 0 else y
    y = y - (int(y + h // 2) - len(frame)) if int(y + h // 2) > len(frame) else y

    # Calc box corners
    min_x = int(x - w // 2)
    max_x = int(x + w // 2)
    min_y = int(y - h // 2)
    max_y = int(y + h // 2)

    patch = frame[min_y:max_y, min_x:max_x]

    return patch, x, y


def update_particle_filter(frame, ref_img, particles, p_weights, sigma, sigma_move_near, sigma_move_far, sigma_ratio,
                           similarity_measure):
    """Run the particle filter logic based on similarity."""
    particles = resample_particles(particles, p_weights)

    h, w = ref_img.shape[:2]

    sims = np.zeros(len(particles))
    new_particles = particles.copy()
    skipped = []
    for i, p in enumerate(particles):
        if i == 0:
            # Keep one copy of best particle
            x, y = p
        elif i < len(particles) * sigma_ratio:
            x = p[0] + int(np.random.normal(0, sigma_move_near))
            y = p[1] + int(np.random.normal(0, sigma_move_near))
        else:
            x = p[0] + int(np.random.normal(0, sigma_move_far))
            y = p[1] + int(np.random.normal(0, sigma_move_far))

        # If particle is off screen, skip
        if (x < 0) or (x > frame.shape[1]) or (y < 0) or (y > frame.shape[0]):
            skipped.append(i)
            continue

        # Get patch and calc probabilities
        patch, x, y = get_patch(frame, w, h, x, y)
        sim, ranking_type = calc_similarity(ref_img, patch, sigma, similarity_measure)
        sims[i] = sim

        new_particles[i] = [x, y]

    # Shift if negative
    if np.min(sims) < 0:
        sims = sims + abs(np.min(sims))

    # Set skipped to worst values
    if ranking_type == 'descending':
        worst = np.min(sims)  # Min to zero, not -1
    else:
        worst = np.max(sims) + 1
    for i in skipped:
        sims[i] = worst

    # Update weights based on similarities
    p_weights = 1.0 * sims / np.sum(sims)
    particles = new_particles

    return particles, p_weights, ranking_type


def reset_particle_filter(frame, ref_img, particles, p_weights, ranking_type):
    """Extract new reference image based on new estimated location."""
    # Best smallest/highest value should depend on comparison metric
    if ranking_type == 'descending':
        idx = np.argsort(p_weights)[::-1][0]
    else:
        idx = np.argsort(p_weights)[0]

    # Select coords based on top particle
    x_best = particles[idx, 0]
    y_best = particles[idx, 1]

    # Make new ref_img, with tracking window included
    h, w = ref_img.shape[:2]

    min_x = int(x_best - w // 2)
    max_x = int(x_best + w // 2)
    min_y = int(y_best - h // 2)
    max_y = int(y_best + h // 2)

    new_particles = particles.copy()
    for i, p in enumerate(particles):
        new_particles[i] = [x_best, y_best]

    # Extract new ref image
    new_ref_img = frame[min_y:max_y, min_x:max_x]

    return new_ref_img, new_particles, (x_best, y_best)


def draw_particles(frame, particles):
    """Draw dots to represent particle locations in the image."""
    for p in particles:
        cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 0, 255), 1)

    return frame


def render(img, ref_img, coords, old_particles, draw_dots, similarity_measure):
    """Display the final image and return keystrokes."""
    h_ref, w_ref = ref_img.shape[:2]
    h, w = img.shape[:2]

    # Make box around ref image
    frame, _, _, _, _ = make_box(img, coords, w_ref, h_ref)

    # Draw particles
    ref_img_save = ref_img.copy()
    if draw_dots:
        _ = draw_particles(img, old_particles)

    # Overlay ref image and text
    img[25:25 + ref_img.shape[0], 5:5 + ref_img.shape[1]] = ref_img_save
    cv2.putText(img, str('Particle filter tracker test, similarity measure: ' + similarity_measure),
                (5, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, [0, 0, 255])
    cv2.putText(img, 'Press the <spacebar> to reset or <ESC> to close the window.',
                (w // 12, h * 19 // 20), cv2.FONT_HERSHEY_PLAIN, 1.0, [0, 0, 255])

    # Show new frame
    cv2.imshow('Camera view', img)
    key = cv2.waitKey(1) & 0xFF

    return key


def run_tracker():
    """Main code to run object tracking using a particle filter."""
    cam = cv2.VideoCapture(0)

    # Particle filter setup
    ref_image_size = 100
    num_particles = 200

    # Choose similarity measure
    similarity_measure = 'MSE_color'

    # Noise parameters motion std/variance
    sigma_move_near = ref_image_size // 2 * 0.10
    sigma_move_far = ref_image_size // 2 * 0.50
    sigma_ratio = 0.25

    # Parameters for MSE probability distribution
    sigma = 5

    # Rendering setup, show particles
    draw_dots = True

    reset = True
    while reset:
        # Get reference image
        ref_img, ref_loc = get_ref_image(cam, similarity_measure, size=ref_image_size)
        if ref_img is None:
            print('WARNING: Frame grab was not successful. Ending.')
            cv2.destroyAllWindows()
            cam.release()
            exit(0)

        h_ref, w_ref = ref_img.shape[:2]

        # Initialize particles
        particles = np.array([ref_loc] * num_particles)
        p_weights = np.ones(num_particles) / num_particles

        while True:
            # Get frame from camera
            ret, img = cam.read()
            if img is None:
                print('WARNING: Frame grab was not successful. Ending.')
                cv2.destroyAllWindows()
                cam.release()
                exit(0)

            h, w, d = img.shape
            img = cv2.flip(img, 1)

            # Process frame and particles
            particles, p_weights, ranking_type = update_particle_filter(img, ref_img, particles, p_weights,
                                                                       sigma, sigma_move_near, sigma_move_far,
                                                                       sigma_ratio, similarity_measure)

            # Update ref image
            old_particles = particles.copy()
            ref_img, particles, coords = reset_particle_filter(img, ref_img, particles, p_weights, ranking_type)

            # Render frame
            key = render(img, ref_img, coords, old_particles, draw_dots, similarity_measure)

            # Keyboard handler
            if key == 27:  # ESC
                reset = False
                break
            if key == ord(' '):  # Reset
                reset = True
                break

    # Release resources
    cv2.destroyAllWindows()
    cam.release()


if __name__ == '__main__':
    run_tracker()
