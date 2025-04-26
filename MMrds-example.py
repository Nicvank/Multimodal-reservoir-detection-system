# 智慧水利系统原型：全流程实现示例

import cv2
import numpy as np
import pandas as pd
import torch
import joblib
from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertModel
import torchvision.models as models
import torchvision.transforms as transforms
import os
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------- 图像预处理模块 ----------------
def preprocess_image(image_stream):
    """读取图像流，判断模糊度，执行归一化和大小调整"""
    image = Image.open(image_stream).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)

    # 判断是否模糊：使用Laplacian方差
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 100:
        logger.warning("图像模糊被丢弃")
        raise ValueError("图像过于模糊，无法处理")

    normalized = image_array / 255.0
    return normalized

# ---------------- 文本日志预处理模块 ----------------
def preprocess_log(file_path):
    """读取日志CSV文件并清洗，输出结构化字典"""
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.to_dict(orient='records')

# ---------------- 图像特征提取模块 ----------------
def extract_image_features(image_array):
    """使用预训练ResNet-50模型提取图像特征向量"""
    model = models.resnet50(pretrained=True)
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(Image.fromarray((image_array * 255).astype(np.uint8))).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze().numpy()

# ---------------- 文本特征提取模块 ----------------
def extract_text_features(text):
    """使用BERT模型提取文本语义向量"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze().numpy()

# ---------------- 故障分类模块 ----------------
def classify_fault(image_feat, text_feat):
    """融合图像与文本特征向量后预测故障类型"""
    model = joblib.load("fault_classifier.pkl")
    fused = np.concatenate([image_feat, text_feat])
    prediction = model.predict([fused])
    confidence = model.predict_proba([fused]).max()
    return prediction[0], confidence

# ---------------- 报告生成模块 ----------------
def generate_report(fault_type, confidence, log_excerpt):
    """基于故障信息生成结构化自然语言报告"""
    timestamp = datetime.utcnow().isoformat()
    report = f"""
    故障检测报告
    ---------------------
    时间: {timestamp}
    故障类型: {fault_type}
    置信度: {confidence:.2%}
    关联日志摘要: {log_excerpt}

    建议处理措施: 请根据《GB/T 30948-2021》规范于48小时内完成处理。
    """
    return report

# ---------------- API 状态监控 ----------------
@app.route('/health', methods=['GET'])
def health_check():
    """返回服务是否健康运行"""
    return jsonify({"status": "ok"}), 200

# ---------------- Flask Web API 接口 ----------------
@app.route('/detect', methods=['POST'])
def detect_fault():
    try:
        image = request.files['image']
        log_text = request.form['log']

        # 1. 预处理并提取特征
        img_array = preprocess_image(image.stream)
        img_feat = extract_image_features(img_array)
        log_feat = extract_text_features(log_text)

        # 2. 故障分类
        label, score = classify_fault(img_feat, log_feat)

        # 3. 报告生成
        report = generate_report(label, score, log_text)

        return jsonify({"label": label, "confidence": score, "report": report})

    except Exception as e:
        logger.error(f"检测失败: {str(e)}")
        return jsonify({"error": str(e)}), 400

# ---------------- 启动服务 ----------------
if __name__ == '__main__':
    logger.info("服务启动中：http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
