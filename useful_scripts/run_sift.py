import cv2
import numpy as np
import itertools
import sys

print  cv2.__version__

def findKeyPoints(img, template, distance=200):

    # # Initiate SIFT detector
    # sift = cv2.SIFT()
    #
    # skp, sd = sift.detectAndCompute.compute(img, None)
    # tkp, td = sift.detectAndCompute.compute(template, None)
    #
    # # # detector = cv2.FeatureDetector_create("SIFT")
    # # detector = cv2.xfeatures2d.SIFT_create()
    # #
    # # # descriptor = cv2.DescriptorExtractor_create("SIFT")
    # #
    # # skp = detector.detect(img)
    # # # skp, sd = descriptor.compute(img, skp)
    # #
    # # tkp = detector.detect(template)
    # # # tkp, td = descriptor.compute(template, tkp)

    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)

    tkp = detector.detect(template)
    tkp, td = descriptor.compute(template, tkp)

    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(sd, flann_params)
    idx, dist = flann.knnSearch(td, 1, params={})
    del flann

    dist = dist[:, 0] / 2500.0
    dist = dist.reshape(-1, ).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    skp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            skp_final.append(skp[i])

    flann = cv2.flann_Index(td, flann_params)
    idx, dist = flann.knnSearch(sd, 1, params={})
    del flann

    dist = dist[:, 0] / 2500.0
    dist = dist.reshape(-1, ).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    tkp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            tkp_final.append(tkp[i])

    return skp_final, tkp_final


def drawKeyPoints(img, template, skp, tkp, num=-1):
    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = (h1 - h2) / 2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[hdif:hdif + h2, :w2] = template
    newimg[:h1, w2:w1 + w2] = img

    maxlen = min(len(skp), len(tkp))
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1] + hdif))
        pt_b = (int(skp[i].pt[0] + w2), int(skp[i].pt[1]))
        cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
    return newimg


def match():
    # img = cv2.imread(sys.argv[1])
    # temp = cv2.imread(sys.argv[2])

    img = cv2.imread('IMG_0098__512.JPG')
    temp = cv2.imread('IMG_0097__512.JPG')

    try:
        dist = int(sys.argv[3])
    except IndexError:
        dist = 200
    try:
        num = int(sys.argv[4])
    except IndexError:
        num = -1
    skp, tkp = findKeyPoints(img, temp, dist)
    newimg = drawKeyPoints(img, temp, skp, tkp, num)
    cv2.imshow("image", newimg)
    cv2.waitKey(0)


match()



# import cv2
# import numpy as np
# import itertools
#
#
# img = cv2.imread('IMG_0098__512.JPG')
# template = cv2.imread('IMG_0097__512.JPG')
#
# detector = cv2.FeatureDetector_create("SIFT")
# descriptor = cv2.DescriptorExtractor_create("SIFT")
#
# skp = detector.detect(img)
# skp, sd = descriptor.compute(img, skp)
#
# tkp = detector.detect(template)
# tkp, td = descriptor.compute(template, tkp)
#
# flann_params = dict(algorithm=1, trees=4)
# flann = cv2.flann_Index(sd, flann_params)
# idx, dist = flann.knnSearch(td, 1, params={})
# del flann
#
# dist = dist[:,0]/2500.0
# dist = dist.reshape(-1,).tolist()
# idx = idx.reshape(-1).tolist()
# indices = range(len(dist))
# indices.sort(key=lambda i: dist[i])
# dist = [dist[i] for i in indices]
# idx = [idx[i] for i in indices]
#
# distance = 10000
# skp_final = []
# for i, dis in itertools.izip(idx, dist):
#     if dis < distance:
#         skp_final.append(skp[i])
#     else:
#         break
#
# h1, w1 = img.shape[:2]
# h2, w2 = template.shape[:2]
# nWidth = w1 + w2
# nHeight = max(h1, h2)
# hdif = (h1 - h2) / 2
# newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
# newimg[hdif:hdif + h2, :w2] = template
# newimg[:h1, w2:w1 + w2] = img
#
# tkp = tkp_final
# skp = skp_fianl
# for i in range(min(len(tkp), len(skp)))
#     pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]+hdif))
#     pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]))
#     cv2.line(newimg, pt_a, pt_b, (255, 0, 0))


# import sys # For debugging only
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# MIN_MATCH_COUNT = 10
#
# img1 = cv2.imread('IMG_0098__512.JPG',0) # queryImage
# img2 = cv2.imread('IMG_0097__512.JPG',0) # trainImage
#
# # Initiate SIFT detector
# sift = cv2.SIFT()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
#
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
#
# flann = cv2.FlannBasedMatcher(index_params, search_params)
#
# matches = flann.knnMatch(des1,des2,k=2)
#
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#
#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#
# else:
#     print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#     matchesMask = None
#
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
#
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#
# plt.imshow(img3, 'gray'),plt.show()