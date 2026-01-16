---
title: "Python数据可视化实战"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/python-data-visualization-practice
date: 2024-01-01
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
