import rclpy
from std_msgs.msg import String,Int32
from rclpy.node import Node
import threading
import cv2
import numpy as np
from scipy.special import softmax
# from scipy.special import expit as sigmoid
from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
from hobot_vio import libsrcampy as srcampy
import time
import logging 
from std_msgs.msg import Header
from std_msgs.msg import Int32MultiArray, MultiArrayDimension


class BaseModel:
    def __init__(
        self,
        model_file: str
        ) -> None:
        # 加载BPU的bin模型, 打印相关参数
        # Load the quantized *.bin model and print its parameters
        try:
            begin_time = time.time()
            self.quantize_model = dnn.load(model_file)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms"%(1000*(time.time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(model_file))
            logger.error("You can download the model file from the following docs: ./models/download.md") 
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        self.model_input_height, self.model_input_weight = self.quantize_model[0].inputs[0].properties.shape[2:4]


    def resizer(self, img: np.ndarray) -> np.ndarray:
        img_h, img_w = img.shape[0], img.shape[1]

        # 计算保持比例的缩放尺寸
        scale = min(self.model_input_weight / img_w, self.model_input_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # 进行缩放
        resized_img = cv2.resize(
            img, 
            (new_w, new_h), 
            interpolation=cv2.INTER_NEAREST
        )

        # 创建目标尺寸的画布
        if len(img.shape) == 3:
            canvas = np.zeros((self.model_input_height, self.model_input_weight, img.shape[2]), dtype=np.uint8)
        else:
            canvas = np.zeros((self.model_input_height, self.model_input_weight), dtype=np.uint8)

        # 计算居中位置
        x_offset = (self.model_input_weight - new_w) // 2
        y_offset = (self.model_input_height - new_h) // 2

        # 将缩放后的图像放入画布
        if len(img.shape) == 3:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized_img
        else:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

        # 更新缩放因子（基于实际缩放后的图像尺寸）
        self.y_scale = img_h / new_h
        self.x_scale = img_w / new_w

        return canvas
    
    def preprocess(self, img: np.ndarray)->np.array:
        """
        Preprocesses an input image to prepare it for model inference.

        Args:
            img (np.ndarray): The input image in BGR format as a NumPy array.

        Returns:
            np.array: The preprocessed image tensor in NCHW format ready for model input.

        Procedure:
            1. Resizes the image to a specified dimension (`input_image_size`) using nearest neighbor interpolation.
            2. Converts the image color space from BGR to RGB.
            3. Transposes the dimensions of the image tensor to channel-first order (CHW).
            4. Adds a batch dimension, thus conforming to the NCHW format expected by many models.
            Note: Normalization to [0, 1] is assumed to be handled elsewhere based on configuration.
        """
        begin_time = time.time()

        #input_tensor = self.resizer(img)
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        # input_tensor = np.array(input_tensor) / 255.0  # yaml文件中已经配置前处理
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.uint8)  # NCHW

        logger.debug("\033[1;31m" + f"pre process time = {1000*(time.time() - begin_time):.2f} ms" + "\033[0m")
        return input_tensor

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image to the NV12 format.

        NV12 is a common video encoding format where the Y component (luminance) is full resolution,
        and the UV components (chrominance) are half-resolution and interleaved. This function first
        converts the BGR image to YUV 4:2:0 planar format, then rearranges the UV components to fit
        the NV12 format.

        Parameters:
        bgr_img (np.ndarray): The input BGR image array.

        Returns:
        np.ndarray: The converted NV12 format image array.
        """
        begin_time = time.time()
        bgr_img = self.resizer(bgr_img)
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed

        logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000*(time.time() - begin_time):.2f} ms" + "\033[0m")
        return nv12


    def forward(self, input_tensor: np.array) -> list[dnn.pyDNNTensor]:
        begin_time = time.time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000*(time.time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs


    def c2numpy(self, outputs) -> list[np.array]:
        begin_time = time.time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time.time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

class YOLO11_Detect(BaseModel):
    def __init__(self, 
                model_file: str, 
                conf: float, 
                iou: float,
                classes_num:int
                ):
        super().__init__(model_file)
        # 将反量化系数准备好, 只需要准备一次
        # prepare the quantize scale, just need to generate once
        self.s_bboxes_scale = self.quantize_model[0].outputs[0].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[2].properties.scale_data[np.newaxis, :]
        logger.info(f"{self.s_bboxes_scale.shape=}, {self.m_bboxes_scale.shape=}, {self.l_bboxes_scale.shape=}")

        # DFL求期望的系数, 只需要生成一次
        # DFL calculates the expected coefficients, which only needs to be generated once.
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")

        # anchors, 只需要生成一次
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                            np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                            np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                            np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)
        logger.info(f"{self.s_anchor.shape = }, {self.m_anchor.shape = }, {self.l_anchor.shape = }")

        # 输入图像大小, 一些阈值, 提前计算好
        self.input_image_size = 640
        self.conf = conf
        self.iou = iou

        self.classes_num=classes_num

        self.conf_inverse = -np.log(1/conf - 1)
        logger.info("iou threshol = %.2f, conf threshol = %.2f"%(iou, conf))
        logger.info("sigmoid_inverse threshol = %.2f"%self.conf_inverse)
    

    def postProcess(self, outputs: list[np.ndarray]) -> tuple[list]:
        begin_time = time.time()
        # reshape
        s_bboxes = outputs[1].reshape(-1, 64)
        m_bboxes = outputs[3].reshape(-1, 64)
        l_bboxes = outputs[5].reshape(-1, 64)
        s_clses = outputs[0].reshape(-1, self.classes_num)
        m_clses = outputs[2].reshape(-1, self.classes_num)
        l_clses = outputs[4].reshape(-1, self.classes_num)


        # classify: 利用numpy向量化操作完成阈值筛选(优化版 2.0)
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # 3个Classify分类分支：Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        # 3个Bounding Box分支：筛选
        s_bboxes_float32 = s_bboxes[s_valid_indices,:]#.astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:]#.astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:]#.astype(np.float32) * self.l_bboxes_scale

        # 3个Bounding Box分支：dist2bbox (ltrb2xyxy)
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        s_anchor_indices = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2])*8

        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        m_anchor_indices = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2])*16

        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        l_anchor_indices = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2])*32

        # 大中小特征层阈值筛选结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

        # nms
        indices = cv2.dnn.NMSBoxes(dbboxes, scores, self.conf, self.iou)

        # 还原到原始的img尺度
        bboxes = dbboxes[indices] # np.array([self.x_scale, self.y_scale, self.x_scale, self.y_scale])
        bboxes = bboxes.astype(np.int32)

        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time.time() - begin_time):.2f} ms" + "\033[0m")

        return ids[indices], scores[indices], bboxes


coco_names = ["peacock","wolf","monkey","elephant","tiger"]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img: np.array, 
                   bbox: tuple[int, int, int, int],
                   score: float, 
                   class_id: int) -> None:
    """
    Draws a detection bounding box and label on the image.

    Parameters:
        img (np.array): The input image.
        bbox (tuple[int, int, int, int]): A tuple containing the bounding box coordinates (x1, y1, x2, y2).
        score (float): The detection score of the object.
        class_id (int): The class ID of the detected object.
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    
    
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

# 初始化摄像头
camera_w=640
camera_h=480#注意不要有重复变量

cam = srcampy.Camera()
cam.open_cam(1, -1, -1, camera_w, camera_h, camera_h, camera_w)  # 设置分辨率,0是cam1，1是cam2
# 实例化
model_path="/home/sunrise/ros2/diansai/ws/src/camera/resource/diansai0801.bin"
conf_thres=0.3
iou_thres=0.5

model = YOLO11_Detect(model_path, conf_thres, iou_thres,5)

img_width = 640  # 图像宽度
img_height = 480  # 图像高度
max_area = img_width * img_height

class PicturePub(Node):
    def __init__(self):
        super().__init__('picture_publisher_node') 
        self.picture_publisher = self.create_publisher(Int32MultiArray, '/pic_cnt', 10)
        # 新增：推理使能标志和锁
        self.enable_subscriber = self.create_subscription(Int32,'/pic_enable',self.enable_callback,10)

        self.enable = 0
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.loop_task)
        self.thread.daemon = True 
        self.thread.start()

    def enable_callback(self, msg):
        """处理/pic_enable话题的回调"""
        with self.lock:
            self.enable = msg.data
            self.get_logger().info(f'推理状态更新: {"启用" if self.enable==1 else "禁用"}')

    def loop_task(self):

        while rclpy.ok():

            with self.lock:
                enable = self.enable

            if enable==0:
                time.sleep(0.01)  # 避免空转消耗CPU
                continue

            # 获取 NV12 图像
            nv12_img = cam.get_img(2, camera_w, camera_h)  # 2 表示 NV12 格式
            if nv12_img is None:
                print("Failed to get image")
                break

            # 转换为 numpy 数组
            nv12_data = np.frombuffer(nv12_img, dtype=np.uint8).reshape(camera_h * 3 // 2,camera_w)

            # NV12 转 BGR（OpenCV 默认格式）
            img = cv2.cvtColor(nv12_data, cv2.COLOR_YUV2BGR_NV12)

            #旋转
            img = cv2.rotate(img, cv2.ROTATE_180)
            

            # 获取图像中心坐标
            # 获取图像尺寸
            height, width = img.shape[:2]
            center_x, center_y = width // 2, height // 2
            # 计算裁剪区域
            crop_width = 480 #0.5m
            crop_height = 480 

            crop_x1 = max(0, center_x - crop_width // 2)
            crop_y1 = max(0, center_y - crop_height // 2)
            crop_x2 = min(width, crop_x1 + crop_width)
            crop_y2 = min(height, crop_y1 + crop_height)
           
            #裁剪画幅
            img=img[crop_y1:crop_y2, crop_x1:crop_x2]
            #缩放
            img = model.resizer(img)

            input_tensor = model.bgr2nv12(img)
            # 推理
            outputs = model.c2numpy(model.forward(input_tensor))
            # 后处理
            ids, scores, bboxes = model.postProcess(outputs)
            # 渲染
            msg = Int32MultiArray()
            msg.layout.dim=[MultiArrayDimension(label='animals', size=5, stride=5)]
            msg.data = [0, 0, 0, 0, 0]
            logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
            for class_id, score, bbox in zip(ids, scores, bboxes):
                x1, y1, x2, y2 = bbox
                msg.data[class_id]+=1
                draw_detection(img, (x1, y1, x2, y2), score, class_id)
                logger.info("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))

            if (sum(msg.data)!=0):
                self.picture_publisher.publish(msg)

            cv2.imshow("VIDEO",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.close_cam()
                cv2.destroyAllWindows()
                break

def main():
    rclpy.init()
    node = PicturePub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cam.close_cam()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()