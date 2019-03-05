import cv2
from matplotlib import pyplot as plt


if __name__ == '__main__':
    orb = cv2.xfeatures2d.SIFT_create()
    hat = cv2.cvtColor(cv2.resize(cv2.imread('imgs/hat2.jpg'), (200, 200)), cv2.COLOR_BGR2GRAY)
    worker = cv2.cvtColor(cv2.imread('imgs/BricklayerUSE_3400020b.jpg'), cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(hat, None)
    kp2, des2 = orb.detectAndCompute(worker, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(hat, kp1, worker, kp2, good, None, flags=2)
    plt.imshow(img3), plt.show()
