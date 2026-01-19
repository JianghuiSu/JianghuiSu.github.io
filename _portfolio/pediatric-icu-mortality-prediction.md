---
title: "基于机器学习的儿童重症监护室患者住院死亡风险预测"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/pediatric-icu-mortality-prediction
date: 2026-01-18
excerpt: "本项目通过分析儿童重症监护室患者的临床数据，构建并比较了逻辑回归、随机森林和SVM三种机器学习模型，用于预测患者住院死亡风险，为临床决策提供支持。"
header:
  teaser: /images/portfolio/pediatric-icu-mortality-prediction/roc_curve_three_models.png
tags:
  - 机器学习
  - 死亡风险预测
  - 儿童重症监护
  - 医疗数据分析
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: SHAP
  - name: Pandas
  - name: Matplotlib
  - name: Seaborn
---

## 项目背景
儿童重症监护室（PICU）患者的病情通常较为严重，死亡风险较高。及时准确地预测患者的死亡风险，有助于临床医生制定更有效的治疗方案，改善患者预后。本项目旨在利用机器学习技术，基于患者的临床数据（如年龄、实验室检查结果等），构建和评估死亡风险预测模型，并通过模型解释技术（SHAP）揭示关键影响因素。

## 核心实现

### 数据预处理
首先对数据进行缺失值填充、异常值处理和特征筛选：
```python
def preprocess_data(data):
    # 缺失值填充（中位数）
    data_no_na = data.fillna(data.median())
    # 异常值处理（IQR法）
    numeric_features = [col for col in data.columns if col != 'HOSPITAL_EXPIRE_FLAG']
    data_clean = handle_outliers(data_no_na, numeric_features)
    # 特征筛选（ANOVA）
    X = data_clean.drop('HOSPITAL_EXPIRE_FLAG', axis=1)
    y = data_clean['HOSPITAL_EXPIRE_FLAG']
    selector = SelectKBest(score_func=f_classif, k='all')
    X_fit = selector.fit(X, y)
    selected_features = p_values[p_values['p_value'] < 0.05]['feature'].tolist()
    X_selected = X[selected_features]
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    # 数据集划分（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, selected_features, scaler, data_clean
```

### 模型构建与调优
构建并调优了逻辑回归、随机森林和SVM三种模型：
```python
def train_and_tune_models(X_train, y_train):
    # 逻辑回归
    lr_params = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
    lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), lr_params, cv=5, scoring='roc_auc')
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    # 随机森林
    rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2,5]}
    rf_grid = GridSearchCV(RandomForestClassifier(class_weight='balanced'), rf_params, cv=5, scoring='roc_auc')
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    # SVM
    svc_params = {'C': [0.1,1,10], 'kernel': ['rbf','linear'], 'gamma': ['scale']}
    svc_grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), svc_params, cv=5, scoring='roc_auc')
    svc_grid.fit(X_train, y_train)
    best_svc = svc_grid.best_estimator_
    return best_lr, best_rf, best_svc
```

### 无监督学习
```python
def unsupervised_analysis(data):
    # 1. 聚类数据预处理（选取实验室特征，避免结局变量干扰）
    cluster_features = [col for col in data.columns if col.startswith('lab_')]
    # 缺失值填充+标准化
    imputer = SimpleImputer(strategy='mean')
    data_cluster_imputed = pd.DataFrame(imputer.fit_transform(data[cluster_features]), columns=cluster_features)
    scaler = StandardScaler()
    data_cluster_scaled = scaler.fit_transform(data_cluster_imputed)
    # 2. 肘部法则确定最佳K值（K=1-10）
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data_cluster_scaled)
        wcss.append(kmeans.inertia_)
    # 绘制肘部图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='darkblue')
    plt.xlabel('簇数 K', fontsize=10)
    plt.ylabel('簇内平方和 (WCSS)', fontsize=10)
    plt.title('KMeans肘部法则图（确定最佳簇数）', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('kmeans_elbow_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("肘部法则图已保存为：kmeans_elbow_plot.png")
    # 3. 最佳K值聚类（基于肘部图选择K=3）
    best_k = 3
    kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(data_cluster_scaled)
    # 4. 聚类评估（轮廓系数）
    silhouette_avg = silhouette_score(data_cluster_scaled, cluster_labels)
    # 5. 聚类与临床结局关联分析
    data_with_cluster = data.copy()
    data_with_cluster['cluster'] = cluster_labels
```

### 模型解释（SHAP）
使用SHAP技术解释模型的预测结果：
```python
def shap_explanation(best_lr, best_rf, X_train, X_test, selected_features):
    import shap
    X_train_df = pd.DataFrame(X_train, columns=selected_features)
    X_test_df = pd.DataFrame(X_test, columns=selected_features)
    # 随机森林SHAP解释
    rf_explainer = shap.TreeExplainer(best_rf)
    rf_shap_values = rf_explainer.shap_values(X_test_df)[1]
    # 逻辑回归SHAP解释
    lr_explainer = shap.LinearExplainer(best_lr, X_train_df)
    lr_shap_values = lr_explainer.shap_values(X_test_df)
    # 绘制SHAP依赖图
    shap.dependence_plot(feat, rf_shap_values, X_test_df, interaction_index=None)
    shap.dependence_plot(feat, lr_shap_values, X_test_df, interaction_index=None)
```

## 分析结果

### 模型性能对比
三种模型的ROC曲线对比：
![ROC曲线](/images/portfolio/pediatric-icu-mortality-prediction/roc_curve_three_models.png)
**分析结论**：随机森林模型的AUC值最高（0.730），其次是SVM（0.712）和逻辑回归（0.679），说明随机森林在预测死亡风险方面表现最优。

### 特征重要性
随机森林模型的特征重要性：
![特征重要性](/images/portfolio/pediatric-icu-mortality-prediction/feature_importance.png)
**分析结论**：lab_5257_min（某实验室指标的最小值）是最重要的特征，其次是lab_5237_min和lab_5235_max，这些特征对患者死亡风险的预测贡献最大。

### SHAP分析
随机森林模型的SHAP依赖图：
![SHAP依赖图](/images/portfolio/pediatric-icu-mortality-prediction/shap_dependence_rf.png)
**分析结论**：SHAP值显示，lab_5257_min的值越高，患者死亡风险越小；而age_month的值与死亡风险的关系较为复杂，需要结合其他特征综合判断。

## 项目附件
为方便复现研究或查看详细内容，以下是本项目的完整代码和分析报告：

| 资源类型       | 文件名                          | 操作链接                                                                 |
|----------------|---------------------------------|--------------------------------------------------------------------------|
| 源数据   | data.xlsx         | [点击下载](/files/data.xlsx) |
| 完整源代码     | full_code.py          | [点击下载](/files/full_code.py) |
| 数据分析报告   | analysis_report.pdf         | [在线预览/下载](/files/analysis_report.pdf) |


### 资源说明
- 源数据为本项目分析所用的数据
- 代码文件包含数据预处理、模型构建、无监督聚类、SHAP 解释等全流程代码
- 数据分析报告内容：包含详细的研究方法、结果解读、局限性分析等全流程细节

---
