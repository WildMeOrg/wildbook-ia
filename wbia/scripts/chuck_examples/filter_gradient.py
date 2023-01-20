# NEED A LIST OF ANNOTATIONS

# aids = ibs.get_valid_aids()
aids = ca_aids


''' ***********************************************
*
*  GRADIENT CHECK --- ELIMINATE LOW-CONTRAST ANNOTS
*
* '''

import cv2 

def gradient_magnitude(image_filepath):
    try:
        image = cv2.imread(image_filepath)
        image = image.astype(np.float32)

        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
    except Exception:
        magnitude = [-1.0]

    result = {
        'sum': np.sum(magnitude),
        'mean': np.mean(magnitude),
        'min': np.min(magnitude),
    }
    return result


#  Access the cached chip files.
chips = ibs.get_annot_chip_fpath(aids)

globals().update(locals())

#  Provide the chips as an argument to the parallel computation
#  of gradient magnitudes.
args = list(zip(chips))
gradient_dicts = list(ut.util_parallel.generate2(
    gradient_magnitude, args, ordered=True
))

# Compute the mean and threshold
gradient_means = ut.take_column(gradient_dicts, 'mean')
gradient_thresh = np.mean(gradient_means) - 2.0 * np.std(gradient_means)  # 2.0 was 1.5

globals().update(locals())

flags = [
    gradient_mean >= gradient_thresh 
    for gradient_mean in gradient_means
]
keep_aids = ut.compress(aids, flags)

delete_aids = list(set(aids) - set(keep_aids))
print(f'Gradient threshold check would remove {len(delete_aids)} out of {len(aids)} annotations')

# Only uncomment this if you really mean it.
# ibs.delete_annots(delete_aids)