from ultralytics import YOLO

# 加载模型
# model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）

# # 使用模型
model.train(data="/usr/src/ultralytics/ultralytics/cfg/datasets/sort.yaml", epochs=100, batch=4,device='0,1')  # 训练模型
# metrics = model.val()  # 在验证集上评估模型性能
# results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
# success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
# print(results)