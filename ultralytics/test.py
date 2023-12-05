from ultralytics import YOLO

def train():
    # train
    # 加载模型
    # model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
    # model = YOLO("/usr/src/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）
    # model = YOLO("/usr/src/ultralytics/ultralytics/yolov8n.pt")  # 加载预训练模型（建议用于训练）,原始网络
    model = YOLO("/usr/src/ultralytics/ultralytics/yolov8n.pt")  # 加载预训练模型（建议用于训练）,原始网络

    # # 使用模型
    model.train(data="/usr/src/ultralytics/ultralytics/cfg/datasets/sort.yaml", epochs=1, batch=4, device='0')  # 训练模型
    # metrics = model.val()  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
    # print(results)

def val():
    # val
    model = YOLO("/usr/src/ultralytics/runs/v1_save/train3/weights/best.pt")
    # 如果不设置数据，它将使用model.pt中的数据集相关yaml文件。
    metrics = model.val()
    metrics = model.val(data='/usr/src/ultralytics/ultralytics/cfg/datasets/sort.yaml')

    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category
    
if __name__ == '__main__':
    train()
    # val()