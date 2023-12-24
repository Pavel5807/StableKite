import cv2
import time
from rknnpool import rknnPoolExecutor
# need to modify image processing functions
from func import myFunc

# cap = cv2.VideoCapture('./video/islandBenchmark.mp4')
cap = cv2.VideoCapture(20)
modelPath = ('./rknnModel/yolov5n_relu_640_640_rockchip_without_postproc_RK3588_i8.rknn')
# number of threads
TPEs = 3
# initialize the rknn pool
pool = rknnPoolExecutor(
    rknnModel=modelPath,
    TPEs=TPEs,
    func=myFunc)

# initialize frames needed for asynchrony
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

frames, loopTime, initTime = 0, time.time(), time.time()

try:
    while (cap.isOpened()):
        frames += 1
        ret, frame = cap.read()
        if not ret:
            break
        pool.put(frame)
        frame, flag = pool.get()
        if flag == False:
            break
        cv2.imshow('test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frames % 30 == 0:
            print("Average frame rate at 30 fps:\t", 30 / (time.time() - loopTime), "frame")
            loopTime = time.time()

    print("Total average frame rate\t", frames / (time.time() - initTime))
except Exception:
    pass
finally:
    print("Total average frame rate\t", frames / (time.time() - initTime))
    # release the cap and rknn thread pools
    cap.release()
    cv2.destroyAllWindows()
    pool.release()
