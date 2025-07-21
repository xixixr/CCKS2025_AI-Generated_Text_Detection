import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":   
    # 读取训练集预测结果
    results_train_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/predictions/train_predictions.jsonl"
    results_train = pd.read_json(results_train_path, lines=True)
    # 利用预测结果和训练集label绘制二分类可视化图

    probabilities = results_train['prediction'].tolist()
    labels = results_train["ground_truth"].tolist()
    # 绘制分布图，横坐标probability，纵坐标为个数，不同样本参照label进行区分
    results_train['label'] = labels
    results_train['probability'] = probabilities
    max_F1,max_p =0,[]
    prediction = []
    for p in tqdm(np.arange(0,0.95,0.0005)):
        predictions = [int(i>=p) for i in probabilities]
        # 计算True Positive, False Positive, True Negative, False Negative
        TP = sum((pred == 1 and label == 1) for pred, label in zip(predictions, labels))
        FP = sum((pred == 1 and label == 0) for pred, label in zip(predictions, labels))
        TN = sum((pred == 0 and label == 0) for pred, label in zip(predictions, labels))
        FN = sum((pred == 0 and label == 1) for pred, label in zip(predictions, labels))
        # 计算F1 score
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1 = 2*precision*recall/(precision+recall)
        if F1>max_F1:
            max_F1 = F1
            max_p = []
            max_p.append(p)
        elif F1 == max_F1:
            max_p.append(p)
    # 打印最大的F1和p
    print(f"max F1 score:{max_F1} , p:{min(max_p),max(max_p)}\n")

    # # 绘制概率分布图
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(data=results_train, x='probability', hue='label', bins=30, kde=True)
    plt.title('Binary Classification Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.grid(True)
    output_path = '/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/images/binary_classification_plot.png'
    plt.savefig(output_path)
    print(f"图像已保存到: {output_path}")

    # 关闭图形以释放内存
    plt.close()