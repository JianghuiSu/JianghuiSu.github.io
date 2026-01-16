---
title: "Python数据可视化与模型评估实战"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/python-data-visualization-practice
date: 2026-01-10
excerpt: "掌握Matplotlib、Seaborn等工具，通过EDA、模型评估、无监督学习及SHAP解释实现数据可视化全流程"
header:
  teaser: /images/portfolio/python-data-visualization-practice/age_distribution.png
tags:
  - 数据可视化
  - 探索性数据分析
  - 模型评估
  - 无监督学习
  - SHAP解释
tech_stack:
  - name: Python
  - name: Matplotlib
  - name: Seaborn
  - name: Scikit-learn
  - name: SHAP
---

## 项目背景  
本项目旨在通过Python实现数据可视化全流程，涵盖探索性数据分析（EDA）、模型评估、无监督学习及模型解释性分析（SHAP）。通过实战练习，掌握Matplotlib、Seaborn等工具的使用，学会为不同场景选择合适的可视化图表。


## 核心实现  

### 1. 探索性数据分析（EDA）  
#### 直方图：年龄分布  
```python  
plt.figure(figsize=(8,5))  
sns.histplot(data=picu_data, x='age_month', kde=True)  
plt.title("年龄分布直方图")  
plt.show()  
```  

### 2. 模型评估可视化  
#### 混淆矩阵热力图  
```python  
def confusion_matrix_plot(y_true, y_pred_prob, threshold=0.5, title='混淆矩阵'):  
    y_pred = (y_pred_prob > threshold).astype(int)  
    cm = confusion_matrix(y_true, y_pred)  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
    plt.xlabel('预测标签')  
    plt.ylabel('真实标签')  
    plt.title(title)  

confusion_matrix_plot(y_test, y_pred_prob)  
plt.show()  
```  

#### ROC曲线  
```python  
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)  
roc_auc = roc_auc_score(y_test, y_pred_prob)  

plt.figure(figsize=(6,5))  
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC={roc_auc:.2f})')  
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')  
plt.xlabel('假阳性率')  
plt.ylabel('真阳性率')  
plt.title(f'ROC曲线 (AUC={roc_auc:.4f})')  
plt.legend()  
plt.show()  
```  

### 3. 无监督学习：肘部法则  
```python  
wcss = []  
for i in range(1,11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  
    kmeans.fit(data_clustering_scaled)  
    wcss.append(kmeans.inertia_)  

plt.figure(figsize=(10,5))  
plt.plot(range(1,11), wcss, marker='o', linestyle='--')  
plt.title('肘部法则')  
plt.xlabel('簇数')  
plt.ylabel('簇内平方和 (WCSS)')  
plt.grid(True)  
plt.show()  
```  

### 4. SHAP模型解释  
#### 蜂群图与条形图  
```python  
# 初始化SHAP解释器  
explainer = shap.Explainer(lr_model, X_train)  
shap_values = explainer(X_train)  

# 蜂群图  
shap.summary_plot(shap_values, X_train)  

# 条形图  
shap.summary_plot(shap_values, X_train, plot_type='bar')  
```  

#### 依赖图  
```python  
shap.dependence_plot('lab_5235_max', shap_values.values, X_train)  
```

## 结果分析  

### 1. EDA结果  
![年龄分布直方图](/images/portfolio/python-data-visualization-practice/age_distribution.png)  
年龄分布呈现右偏，多数样本集中在低年龄区间。  

![实验室指标箱线图](/images/portfolio/python-data-visualization-practice/boxplot_lab_indices.png)  
不同结局下，实验室指标（如lab_5235_max）分布存在显著差异，可作为预测特征。  

### 2. 模型评估  
![混淆矩阵](/images/portfolio/python-data-visualization-practice/confusion_matrix.png)  
模型在测试集上的混淆矩阵显示，真阳性率与真阴性率表现均衡。  

![ROC曲线](/images/portfolio/python-data-visualization-practice/roc_curve.png)  
ROC曲线下面积（AUC）为0.6997，表明模型的区分能力一般。  

### 3. 无监督学习  
![肘部法则图](/images/portfolio/python-data-visualization-practice/elbow_method.png)  
肘部出现在k=3附近，提示数据可分为3个簇。  

### 4. SHAP解释  
![SHAP蜂群图](/images/portfolio/python-data-visualization-practice/shap_beeswarm.png)  
lab_5257_min是影响预测结果的关键特征，值越高死亡风险越大。  

![SHAP依赖图](/images/portfolio/python-data-visualization-practice/shap_dependence_lab5235.png)  
lab_5235_max与SHAP值呈正相关，验证了其对死亡风险的正向影响。  
