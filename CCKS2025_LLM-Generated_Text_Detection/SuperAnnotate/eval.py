import torch
from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
from generated_text_detector.utils.preprocessing import preprocessing_text
from transformers import AutoTokenizer
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import tqdm
import json 

class AIDetector:
    def __init__(self, model_fine, device=None):
        # 从本地加载模型
        model_path = "/data/workspace/xiarui/tianchi_LMTextDetect/ai-detector"
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.model = RobertaClassifier.from_pretrained(model_path, local_files_only=True)
        self.model = self.model.bfloat16()  # 使用bfloat16精度
        # 检查模型是否加载成功
        if self.model is None:
            raise ValueError("Failed to load the model. Please check the model path or files.")
        # 加载微调后的模型权重
        state_dict = torch.load(model_fine)
        
        # 处理多卡训练保存的模型
        if all(k.startswith('module.') for k in state_dict.keys()):
            # 如果是在多卡环境下保存的模型，需要去掉'module.'前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        device_ids = [1]
        self.device = device if device else torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text, threshold=0.5195):
        text = preprocessing_text(text)
        
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='longest',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            _, logits = self.model(**tokens)
            proba = logits.squeeze(1).item()
            # proba = F.sigmoid(logits).squeeze(1).item()
        
        prediction = 1 if proba >= threshold else 0
        return {
            "probability": proba,
            "prediction": prediction,
            "label": "AI-generated" if prediction == 1 else "Human-written"
        }

# 使用示例
if __name__ == "__main__":
    # 训练集上最大的p
    max_p = 0.82
    # 加载微调后的模型
    # 加载训练集和测试集
    train_text_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/train.jsonl"
    train_text = pd.read_json(train_text_path, lines=True)
    train_text_label = train_text['label'].tolist()
    train_text = train_text['text'].tolist()
    
    
    test_text_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/test.jsonl"
    test_text = pd.read_json(test_text_path, lines=True)
    test_text = test_text['text'].tolist()
    model_fine = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/saved_models/model_7619.pt"
    detector = AIDetector(model_fine)
    
    # 示例文本
    text_example = "On May 28, XX Group, a leading domestic green technology enterprise, held the \"Technology for New Horizons · Green Motion for the Future\" technology conference in Beijing, officially launching its self-developed \"Ruineng 2.0\" high-efficiency energy-saving system and supporting solutions, injecting innovative momentum into the low-carbon transformation of industrial sectors. According to the conference, the \"Ruineng 2.0\" system achieves dynamic optimization of industrial equipment energy consumption through the deep integration of an intelligent energy efficiency management platform and new energy storage technologies; third-party testing shows that after application in pilot enterprises, this technology has reduced comprehensive energy consumption by an average of 18.7 and achieved an annual carbon emission reduction of over 2,000 tons. The simultaneously released \"Photovoltaic-Storage Integration\" park solution combines solar power generation, energy storage systems, and intelligent microgrids to increase the utilization rate of renewable energy in parks to over 65%. \"We have always regarded technological innovation as the core driving force for green development,\" said Li Hua, CTO of XX Group, adding that the released technological achievements have been validated in five major industries including metallurgy and chemical engineering, and the company plans to invest 500 million yuan in low-carbon technology research and development over the next three years to help 100 key enterprises complete energy-saving transformations. Industry experts note that against the backdrop of continuous progress toward the \"double carbon\" goals, the implementation of these technologies will provide a replicable innovative model for the green transformation of traditional industries and promote the comprehensive green and low-carbon transformation of economic and social development."
    # 预测示例文本
    example_result = detector.predict(text_example,max_p)
    print(f"Example Prediction: {example_result['prediction']}, Probability: {example_result['probability']:.4f}, Label: {example_result['label']}")
    # 示例文本
    text_example = "I have a dog,which is white and cute."
    # 预测示例文本
    example_result = detector.predict(text_example,max_p)
    print(f"Example Prediction: {example_result['prediction']}, Probability: {example_result['probability']:.4f}, Label: {example_result['label']}")
    
    # 训练集上进行预测
    results_train = []
    for i in tqdm(range(len(train_text)),ncols=100, desc="Predicting"):
        result = detector.predict(train_text[i])
        results_train.append(result)
    # 保存训练集预测结果到文件
    output_dir = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "train_predictions.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results_train:
            json_line = json.dumps(result, ensure_ascii=False) # 转换为符合json格式的文件
            f.write(f"{json_line}\n")#替换为双引号
            
   
    #测试集上进行预测
    results = []
    for i in tqdm(range(len(test_text)),ncols=100, desc="Predicting"):
        result = detector.predict(test_text[i],threshold=max_p)
        results.append(result)
    # 将结果的prediction按行写入submit.txt
    submit_dir = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/submit_A"
    with open(os.path.join(submit_dir, "submit.txt"), "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"{result['prediction']}\n")