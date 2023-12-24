#以下代码改自https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import traceback
import cv2
import numpy as np
import matplotlib.pyplot as plt

OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.25, 0.45, 640
IMG_HEIGHT = 736
IMG_WIDTH = 1280
IM_SIZE = (736, 1280)
IMG_SIZE = (1280, 736)

# IMG_HEIGHT = 640
# IMG_WIDTH = 640
# IM_SIZE = (640, 640)
# IMG_SIZE = (640, 640)

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

CLASSES = ("person", "tank")


def show_image(image):
    # _img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filter_boxes(boxes, box_confidences, box_class_probs, args):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    # Distribution Focal Loss (DFL)
    # import torch
    # x = torch.tensor(position)
    # print(position.shape)
    x = np.array(position)
    # print(x.shape)

    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = softmax(y)
    acc_metrix = np.array(range(mc)).astype(np.float32).reshape(1,1,mc,1,1)
    y = np.sum(y*acc_metrix, axis=2)
    return y

def softmax(x):
    x_max = np.amax(x, axis=2, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=2, keepdims=True)
    # return np.exp(x)/np.sum(np.exp(x), axis=2)

def box_process(position, anchors, args):
    # print(position.shape)
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data, anchors, args):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i], None, args))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf, args)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    
    
def initRKNN(model_path):
	# rknnModel="./rknnModel/yolov5s.rknn" #add path
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    return rknn_lite
	

class yolov8:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(self.model_path)
        if ret != 0:
            print("Load RKNN rknnModel failed")
            exit(ret)
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print("Init runtime environment failed")
            exit(ret)


    def draw(cls, image, boxes, scores, classes):
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            # print('class: {}, score: {}'.format(CLASSES[cl], score))
            # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
            top = int(top)
            left = int(left)

            cv2.rectangle(image, (top, left), (int(right), int(bottom)), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        return image
            

    def run(self, input):
        IMG = cv2.resize(input, IMG_SIZE)
        
        outputs = self.rknn_lite.inference(inputs=[IMG])
        anchors = [[[1.0,1.0]]]*3

        boxes, classes, scores = post_process(outputs, anchors, None)
        # IMG = cv2.cvtColor(IMG, cv2.COLOR_RGB2BGR)
        # if boxes is not None:
        #     draw(IMG, boxes, scores, classes)
        # return IMG
        if boxes is None:
            return None, None, None
        class_names = []
        for cl in classes:
            class_names.append(CLASSES[cl])
        return classes, scores, boxes
    
    def __del__(self):
        self.rknn_lite.release()


def yolov5_(model_path, IMG):
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    # 等比例缩放
    # IMG = letterbox(IMG)
    # 强制放缩
    IMG = cv2.resize(IMG, (IMG_SIZE, IMG_SIZE))
    
    rknn_lite = initRKNN(model_path)
    outputs = rknn_lite.inference(inputs=[IMG])

    input0_data = outputs[0].reshape([3, -1]+list(outputs[0].shape[-2:]))
    input1_data = outputs[1].reshape([3, -1]+list(outputs[1].shape[-2:]))
    input2_data = outputs[2].reshape([3, -1]+list(outputs[2].shape[-2:]))

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    boxes, classes, scores = yolov5_post_process(input_data)
    return classes, scores, boxes[0], boxes[1], boxes[2], boxes[3]

    IMG = cv2.cvtColor(IMG, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        draw(IMG, boxes, scores, classes)
    return IMG
    


if __name__ == "__main__":
    yolo = yolov8('/home/orangepi/ros2_ws/src/yolov5_ros/yolov5_ros/rknn/new/yolov8n_1280_736_b16_e500_relu_rockchip_rknnopt_cut_RK3588_i8.rknn')
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('/home/orangepi/Projects//tanks.mp4')

    try:
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            print(frame.shape)
            out = yolo.run(frame)
            cv2.imshow('test', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception:
        traceback.print_exc()
    finally:
        # release the cap and rknn thread pools
        cap.release()
        cv2.destroyAllWindows()
