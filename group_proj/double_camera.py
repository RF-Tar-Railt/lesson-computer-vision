import cv2
import numpy as np


left_camera_matrix = np.array(
    [
        #                      there should be 0 if calibrate is very correct
        [4.700848502429587e+02, 0, 5.440618993273372e+02],
        [0, 4.661448130257676e+02, 3.337074417612100e+02],
        [0, 0, 1],
    ], dtype='float64'
)

left_distortion = np.array(
    [
        [
            -0.060004210421966,  # -0.013174360936415,  # k1
            0.005563217117779,  # 0.217618268684277,  # k2
            0.001912597679823,  # -0.002462636541929,  # p1
            0.002983089771792,  # -0.003152044338437,  # p2
            2.522987648938966e-04,  # -0.530544688052719,  # k3
        ]
    ], dtype='float64'
)

right_camera_matrix = np.array(
    [
        #                      there should be 0 if calibrate is very correct
        [4.651381033054275e+02, 0, 5.453743178709904e+02],
        [0, 4.601398341497697e+02, 3.409474852228430e+02],
        [0, 0, 1],
    ], dtype='float64'
)

right_distortion = np.array(
    [
        [
            -0.058759049853852,  # -0.010701696628130,  # k1
            0.005998962391883,  # 0.213264008998207,  # k2
            0.001981457940990,  # -0.002422861971945,  # p1
            0.003174101066657,  # -0.002357193114398,  # p2
            0.001130768555751,  # -0.540114506676907,  # k3
        ]
    ], dtype='float64'
)

rec = np.array(
    [
        [0.999763224899467, -5.720570423647862e-04, -0.021752399622971],
        [5.463956709250523e-04, 0.999999147868492, -0.001185628128796],
        [0.021753059333987, 0.001173461984591, 0.999762685539215],
    ]
)

R = cv2.Rodrigues(rec)[0]

# T = np.array([[-61.507817546611020], [-1.375821012862682], [-3.341776157686859]])
T = np.array([[-61.101700017582100], [1.753679016270766], [-1.741524521448996]])
size = (640, 480)  # 图像尺寸

# 进行立体更正, bouguet标定方法
Rl, Rr, Pl, Pr, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    left_camera_matrix,
    left_distortion,
    right_camera_matrix,
    right_distortion,
    size,
    R,
    T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0,
)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, Rl, Pl, size, cv2.CV_32FC1
)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, Rr, Pr, size, cv2.CV_32FC1
)

BM_stereo = cv2.StereoSGBM.create(numDisparities=16, blockSize=9)

disp = np.zeros((480, 640), dtype=np.float32)
threeD: np.ndarray


def callback(e, x, y, f, p):
    global threeD
    global disp
    if e == cv2.EVENT_LBUTTONDOWN:
        #for _x, _y in [(x, y), (x + 10, y + 10), (x - 10, y - 10), (x + 10, y - 10), (x - 10, y + 10)]:
        num = disp[y][x]

        print(num)
        _tmp = abs(T[0]) * Q[2][3] / abs(num)
        print("distance: ", _tmp / 10, "cm")

            # point3 = threeD[_y][_x]
            # print("world coordinate: ")
            # print("x: ", point3[0], "y: ", point3[1], "z: ", point3[2])
            # d = point3[0] ** 2 + point3[1] ** 2 + point3[2] ** 2
            # d **= 0.5
            # d /= 10
            # print("distance: ", d, "cm")
        point3 = threeD[y][x]
        print("world coordinate: ")
        print("x: ", point3[0], "y: ", point3[1], "z: ", point3[2])
        d = (point3[0] ** 2) + (point3[1] ** 2) + (point3[2] ** 2)
        d **= 0.5
        d /= 10
        d *= (100/120)
        print("distance: ", d, "cm")


def BM_update(val=0):
    global SGBM_num
    global SGBM_blockSize
    global BM_stereo
    global threeD
    global disp
    SGBM_blockSize = cv2.getTrackbarPos("blockSize", "SGNM_disparity")

    BM_stereo.setBlockSize(2 * SGBM_blockSize + 5)
    # BM_stereo.setROI1(validPixROI1)
    # BM_stereo.setROI2(validPixROI2)
    BM_stereo.setP1(8 * 3 * SGBM_blockSize ** 2)
    BM_stereo.setP2(32 * 3 * SGBM_blockSize ** 2)
    BM_stereo.setPreFilterCap(31)
    BM_stereo.setMinDisparity(0)
    SGBM_num = cv2.getTrackbarPos("num_disp", "SGNM_disparity")
    num_disp = SGBM_num * 16 + 16
    BM_stereo.setNumDisparities(num_disp)
    # BM_stereo.setTextureThreshold(10)
    BM_stereo.setUniquenessRatio(cv2.getTrackbarPos("unique_Ratio", "SGNM_disparity"))
    BM_stereo.setSpeckleWindowSize(
        100  # cv2.getTrackbarPos("spec_WinSize", "SGNM_disparity")
    )
    BM_stereo.setSpeckleRange(32)  # cv2.getTrackbarPos("spec_Range", "SGNM_disparity")
    BM_stereo.setDisp12MaxDiff(-1)
    left_image_down = cv2.pyrDown(imgL)
    right_image_down = cv2.pyrDown(imgR)
    factor = imgL.shape[1] / left_image_down.shape[1]
    disparity_left_half = BM_stereo.compute(left_image_down, right_image_down)
    disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
    disparity_left = factor * disparity_left
    disparity_left = disparity_left.astype(np.float32)
    disp = disparity_left / 16.0
    disp = cv2.medianBlur(disp, 5)
    threeD = cv2.reprojectImageTo3D(disparity_left, Q, handleMissingValues=False)
    threeD = threeD * 16
    cv2.imshow("SGNM_disparity", disp / num_disp)
    # disp_color = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # disp_color = cv2.applyColorMap(disp_color.astype(np.uint8), cv2.COLORMAP_JET)


if __name__ == "__main__":
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    SGBM_blockSize = 7  # 一个匹配块的大小,大于1的奇数
    SGBM_num = 6
    uniquenessRatio = 2
    # 创建窗口
    cv2.namedWindow("SGNM_disparity")
    cv2.createTrackbar("blockSize", "SGNM_disparity", SGBM_blockSize, 8, BM_update)
    cv2.createTrackbar("num_disp", "SGNM_disparity", SGBM_num, 16, BM_update)
    cv2.createTrackbar(
        "unique_Ratio", "SGNM_disparity", uniquenessRatio, 50, BM_update
    )
    while True:
        ret, frame = cap.read()
        frame1 = frame[:, 0:640, :]
        frame2 = frame[:, 640:1280, :]
        imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        imgL = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
        imgR = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)
        size = (imgL.shape[1], imgL.shape[0])
        cv2.imshow("left", imgL)
        cv2.imshow("right", imgR)
        cv2.setMouseCallback("left", callback, None)
        BM_update()
        key = cv2.waitKey(1)
        if key & 0xFF in (27, ord("q")):
            break
        elif key & 0xFF == ord("r"):
            print("Resetting")

    cv2.destroyAllWindows()  # 关闭所有地窗口
    cap.release()  # 释放摄像头
