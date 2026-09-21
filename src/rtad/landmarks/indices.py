"""FaceMesh landmark indices used for feature extraction (468 mesh + 10 iris points)."""

# Eye contour points in EAR order: [outer, top1, top2, inner, bottom2, bottom1]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Iris points (FaceLandmarker always outputs them)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

# Mouth: [left corner, top1, top2, right corner, bottom2, bottom1]
MOUTH = [61, 81, 311, 291, 402, 178]

# Points for solvePnP head pose: nose tip, chin, left eye corner, right eye corner,
# left mouth corner, right mouth corner
HEAD_POSE = [1, 152, 263, 33, 291, 61]
