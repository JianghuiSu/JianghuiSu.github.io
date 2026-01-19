# 导入所有所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1. 修正calibration_curve的导入路径，新增silhouette_score
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_score, f1_score, brier_score_loss,
    silhouette_score  # 用于聚类评估的轮廓系数
)

from sklearn.calibration import calibration_curve  # 单独导入calibration_curve
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")
# Defer importing shap to shap_explanation() and disable Torch backend there
# to avoid attempting to load torch/fbgemm.dll on Windows.
import os
os.environ["SHAP_DISABLE_TORCH"] = "1"  # 在导入 shap 或启动并行任务之前设置，防止子进程加载 torch 的 shm.dll 导致崩溃


# 3. 手动实现特异度计算函数（保留，正确）
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()  # 拆解混淆矩阵
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    return specificity

# 4. 中文字体设置（无错误）
plt.rcParams["font.sans-serif"] = 'SimHei'
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

# ===================== 1. 数据读取与基础概览 =====================
def load_and_overview(path):
    """读取数据并输出基础信息"""
    data = pd.read_excel(path)
    print("="*50)
    print("数据基础概览")
    print("="*50)
    print(f"数据维度 (样本数, 特征数): {data.shape}")
    print("\n缺失值统计（前10列）:")
    print(data.isnull().sum().head(10))
    print("\n结局变量分布:")
    print(data['HOSPITAL_EXPIRE_FLAG'].value_counts())
    print(f"死亡样本占比: {data['HOSPITAL_EXPIRE_FLAG'].mean():.3f}")
    return data

# ===================== 1.1 数据探索 - 描述性统计与可视化（缺失值填充前） =====================
def exploratory_analysis(data):
    # """
    # 缺失值填充前的探索性分析：
    # 1. 计算数值特征的均值、中位数、方差
    # 2. 绘制6个指定数值特征的直方图（age_month + 5个lab特征）
    # 3. 按结局变量分组绘制6个特征的箱线图
    # """
    # print("\n" + "="*50)
    # print("数据探索 - 描述性统计与可视化（缺失值填充前）")
    # print("="*50)
    
    # 步骤1：筛选数值特征（排除结局变量）
    # 指定6个分析特征：age_month + 5个lab指标（覆盖所有数值特征）
    numeric_features = [col for col in data.columns if col != 'HOSPITAL_EXPIRE_FLAG']
    target_feature = 'HOSPITAL_EXPIRE_FLAG'
    print(f"分析的6个数值特征：{numeric_features}")
    
    # 处理特征数量≠6的情况，确保子图不超出范围
    n_features = len(numeric_features)
    if n_features > 6:
        print(f"警告：数值特征数量({n_features})超过6个，自动选取前6个特征分析")
        numeric_features = numeric_features[:6]  # 取前6个
    elif n_features < 6:
        print(f"警告：数值特征数量({n_features})不足6个，使用全部特征分析")
    print(f"最终分析的数值特征：{numeric_features}")
    n_features = len(numeric_features)  # 更新为实际分析的数量


    # 步骤2：计算均值、中位数、方差（缺失值填充前）
    # 排除NaN值计算统计量，保留3位小数
    desc_stats = data[numeric_features].agg(
        ['mean', 'median', 'var']  # 均值、中位数、方差
    ).round(3)
    # 转置表格（特征为行，统计量为列），便于阅读
    desc_stats = desc_stats.T
    desc_stats.columns = ['均值', '中位数', '方差']  # 中文列名
    print("\n缺失值填充前 - 数值特征描述性统计：")
    print(desc_stats)
    # 保存统计结果到CSV（便于报告引用）
    desc_stats.to_csv('descriptive_stats_before_imputation.csv', encoding='utf-8-sig')
    print("\n统计结果已保存为：descriptive_stats_before_imputation.csv")
    
    # 步骤3：绘制6个特征的直方图（观察分布形态）
    import math  # 若文件开头未导入，需在此处导入
    n_cols = 3  # 固定3列
    n_rows = math.ceil(n_features / n_cols)  # 自动计算行数（如5个特征→2行，7个→3行）
    plt.figure(figsize=(15, 5 * n_rows))  # 按行数调整画布高度
    for i, feat in enumerate(numeric_features, 1):
        plt.subplot(2, 3, i)  # 2行3列子图布局（刚好放6个特征）
        # 绘制直方图，bins=30（分组数），边缘色黑色，透明度0.7
        sns.histplot(data=data, x=feat, bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'{feat} 分布直方图（缺失值填充前）', fontsize=10)
        plt.xlabel(feat, fontsize=8)
        plt.ylabel('频数', fontsize=8)
        plt.tight_layout()  # 自动调整子图间距，避免重叠
    # 保存直方图
    plt.savefig('feature_histograms_before_imputation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("直方图已保存为：feature_histograms_before_imputation.png")
    
    # 步骤4：按结局变量分组绘制6个特征的箱线图（探索组间差异）
    plt.figure(figsize=(15, 5 * n_rows))
    for i, feat in enumerate(numeric_features, 1):
        plt.subplot(n_rows, n_cols, i)
        # 按结局变量分组（0=存活，1=死亡），颜色区分两组
        sns.boxplot(
            data=data, x=target_feature, y=feat,
            palette=['lightblue', 'lightcoral'],  # 改为列表以避免键类型不匹配
            showfliers=False
        )
        # 替换x轴标签（0→存活，1→死亡），更直观
        plt.xticks([0, 1], ['存活(0)', '死亡(1)'], fontsize=8)
        plt.title(f'{feat} 按结局分组箱线图', fontsize=10)
        plt.xlabel('患者结局', fontsize=8)
        plt.ylabel(feat, fontsize=8)
        plt.tight_layout()
    # 保存箱线图
    plt.savefig('grouped_boxplots_by_outcome.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("分组箱线图已保存为：grouped_boxplots_by_outcome.png")

# ===================== 2. 数据预处理 =====================
def preprocess_data(data):
    """数据预处理：缺失值填充、异常值处理、特征筛选、标准化"""
    print("\n" + "="*50)
    print("数据预处理")
    print("="*50)
    
    # 2.1 缺失值填充（中位数）
    data_no_na = data.fillna(data.median())
    print(f"填充后缺失值总数: {data_no_na.isnull().sum().sum()}")
    
    # 2.2 异常值处理（IQR法）
    def handle_outliers(df, features):
        df_clean = df.copy()
        for col in features:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].apply(
                    lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x
                )
        return df_clean
    
    numeric_features = [col for col in data.columns if col != 'HOSPITAL_EXPIRE_FLAG']
    data_clean = handle_outliers(data_no_na, numeric_features)
    print("异常值处理完成（IQR法）")
    
    # 2.3 特征筛选（ANOVA）
    X = data_clean.drop('HOSPITAL_EXPIRE_FLAG', axis=1)
    y = data_clean['HOSPITAL_EXPIRE_FLAG']
    
    selector = SelectKBest(score_func=f_classif, k='all')
    X_fit = selector.fit(X, y)
    p_values = pd.DataFrame({'feature': X.columns, 'p_value': X_fit.pvalues_})
    selected_features = p_values[p_values['p_value'] < 0.05]['feature'].tolist()
    X_selected = X[selected_features]
    print(f"ANOVA筛选后保留特征数: {len(selected_features)}")
    print(f"筛选后的特征: {selected_features}")
    
    # 2.4 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # 2.5 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y  # 分层抽样解决不平衡
    )
    print(f"\n训练集维度: {X_train.shape}, 测试集维度: {X_test.shape}")
    print(f"训练集死亡占比: {y_train.mean():.3f}, 测试集死亡占比: {y_test.mean():.3f}")
    
    # 关键修改：补充data_clean的返回，使返回值数量为7个
    return X_train, X_test, y_train, y_test, selected_features, scaler, data_clean


# ===================== 3. 统计分析 =====================
def statistical_analysis(data_clean, selected_features):
    """统计分析：分组统计、组间检验、相关性热力图"""
    print("\n" + "="*50)
    print("统计分析")
    print("="*50)
    
    # 新增：检查筛选后特征是否为空
    if len(selected_features) == 0:
        print("警告：ANOVA筛选后无有效特征，跳过统计分析")
        return  # 直接返回，避免后续报错
    
    # 3.1 分组描述性统计
    grouped = data_clean.groupby('HOSPITAL_EXPIRE_FLAG')[selected_features].agg(['mean', 'std', 'median'])
    print("\n分组描述性统计（存活=0，死亡=1）:")
    print(grouped.round(3).head())
    
    # 3.2 组间t检验
    print("\n组间差异检验（t检验）:")
    for feat in selected_features[:6]:  # 展示前6个特征
        group0 = data_clean[data_clean['HOSPITAL_EXPIRE_FLAG'] == 0][feat]
        group1 = data_clean[data_clean['HOSPITAL_EXPIRE_FLAG'] == 1][feat]
        t_stat, p_val = ttest_ind(group0, group1, equal_var=False)
        print(f"{feat}: t={t_stat:.3f}, p={p_val:.3f}")
    
    # 3.3 相关性热力图
    plt.figure(figsize=(10, 8))
    corr_matrix = data_clean[selected_features].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('相关性热力图', fontsize=12)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n相关性热力图已保存为 correlation_heatmap.png")


# ===================== 4. 无监督学习分析（KMeans聚类） =====================
def unsupervised_analysis(data):
    print("\n" + "="*50)
    print("无监督学习分析（KMeans聚类）")
    print("="*50)
    
    # 1. 聚类数据预处理（选取实验室特征，避免结局变量干扰）
    cluster_features = [col for col in data.columns if col.startswith('lab_')]
    print(f"用于聚类的实验室特征：{cluster_features}")
    
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
    print(f"\n最佳簇数 K={best_k}")
    print(f"轮廓系数: {silhouette_avg:.4f}（>0.5表示聚类结构合理）")
    
    # 5. 聚类与临床结局关联分析
    data_with_cluster = data.copy()
    data_with_cluster['cluster'] = cluster_labels
    cluster_outcome = data_with_cluster.groupby('cluster')['HOSPITAL_EXPIRE_FLAG'].agg(['count', 'mean']).round(3)
    cluster_outcome.columns = ['样本数', '死亡占比']
    print("\n各聚类的临床结局分布：")
    print(cluster_outcome)
    
    # 绘制聚类-死亡占比条形图
    plt.figure(figsize=(8, 5))
    sns.barplot(x=cluster_outcome.index, y='死亡占比', data=cluster_outcome, palette='Set2')
    plt.xlabel('聚类编号', fontsize=10)
    plt.ylabel('死亡占比', fontsize=10)
    plt.title(f'K={best_k}时各聚类的患者死亡占比', fontsize=12)
    # 添加数值标签
    for i, v in enumerate(cluster_outcome['死亡占比']):
        plt.text(i, v + 0.005, f'{v:.1%}', ha='center', fontsize=9)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('cluster_outcome_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("聚类-死亡占比图已保存为：cluster_outcome_ratio.png")
    
    return cluster_labels, silhouette_avg

# ===================== 5. 模型构建与调优 =====================
def train_and_tune_models(X_train, y_train):
    print("\n" + "="*50)
    print("模型构建与调优")
    print("="*50)
    
    # 5.1 逻辑回归
    lr_params = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        lr_params, cv=5, scoring='roc_auc', n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    print(f"\n逻辑回归最优参数: {lr_grid.best_params_}")
    print(f"逻辑回归交叉验证AUC: {lr_grid.best_score_:.3f}")
    
    # 5.2 随机森林
    rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        rf_params, cv=5, scoring='roc_auc', n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"\n随机森林最优参数: {rf_grid.best_params_}")
    print(f"随机森林交叉验证AUC: {rf_grid.best_score_:.3f}")
    
    # 5.3 支持向量机（SVM）- 新增
    svc_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],  # 径向基核与线性核
        'gamma': ['scale']  # 自动计算gamma
    }
    svc_grid = GridSearchCV(
        SVC(probability=True, random_state=42, class_weight='balanced'),  # probability=True用于AUC计算
        svc_params, cv=5, scoring='roc_auc', n_jobs=-1
    )
    svc_grid.fit(X_train, y_train)
    best_svc = svc_grid.best_estimator_
    print(f"\n支持向量机（SVM）最优参数: {svc_grid.best_params_}")
    print(f"SVM交叉验证AUC: {svc_grid.best_score_:.3f}")
    
    return best_lr, best_rf, best_svc  # 新增best_svc返回

# ===================== 6. 模型评估与可视化（含SVM评估） =====================
def evaluate_and_visualize(best_lr, best_rf, best_svc, X_test, y_test, selected_features):
    print("\n" + "="*50)
    print("模型评估与可视化")
    print("="*50)
    
    # 评估函数
    def evaluate_model(model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        results = {
            '模型': model_name,
            '准确率': accuracy_score(y_test, y_pred),
            '召回率': recall_score(y_test, y_pred),
            '精确率': precision_score(y_test, y_pred, zero_division=0),
            '特异度': specificity_score(y_test, y_pred),
            'F1分数': f1_score(y_test, y_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, y_pred_prob),
            'Brier分数': brier_score_loss(y_test, y_pred_prob)
        }
        print(f"\n{model_name} 评估结果:")
        for k, v in results.items():
            if k != '模型':
                print(f"{k}: {v:.4f}")
        return results, y_pred_prob
    
    # 评估三个模型
    lr_results, lr_prob = evaluate_model(best_lr, X_test, y_test, "调优逻辑回归")
    rf_results, rf_prob = evaluate_model(best_rf, X_test, y_test, "调优随机森林")
    svc_results, svc_prob = evaluate_model(best_svc, X_test, y_test, "调优支持向量机（SVM）")  # 新增SVM评估
    
    # ROC曲线（含三个模型）
    plt.figure(figsize=(8, 6))
    # 逻辑回归
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    plt.plot(fpr_lr, tpr_lr, label=f'逻辑回归 (AUC={lr_results["AUC"]:.3f})', linewidth=2)
    # 随机森林
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
    plt.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC={rf_results["AUC"]:.3f})', linewidth=2)
    # SVM
    fpr_svc, tpr_svc, _ = roc_curve(y_test, svc_prob)
    plt.plot(fpr_svc, tpr_svc, label=f'SVM (AUC={svc_results["AUC"]:.3f})', linewidth=2)
    # 随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('假阳性率', fontsize=10)
    plt.ylabel('真阳性率', fontsize=10)
    plt.title('三种模型ROC曲线对比', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve_three_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n三模型ROC曲线已保存为：roc_curve_three_models.png")
    
    # 混淆矩阵（随机森林，最优模型）
    plt.figure(figsize=(6, 5))
    y_pred_rf = best_rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['存活', '死亡'], yticklabels=['存活', '死亡'])
    plt.xlabel('预测标签', fontsize=10)
    plt.ylabel('真实标签', fontsize=10)
    plt.title('随机森林混淆矩阵', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("混淆矩阵已保存为：confusion_matrix.png")
    
    # 特征重要性（随机森林）
    feature_importance = pd.DataFrame({
        'feature': selected_features, 'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('随机森林Top6特征重要性', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("特征重要性图已保存为：feature_importance.png")
    
    # 保存评估结果表格
    results_df = pd.DataFrame([lr_results, rf_results, svc_results])
    results_df.to_csv('model_results_three_models.csv', index=False, encoding='utf-8-sig')
    print("\n三模型评估结果已保存为：model_results_three_models.csv")

# ===================== 7. 模型解释性分析（SHAP） =====================

def shap_explanation(best_lr, best_rf, X_train, X_test, selected_features):
    print("\n" + "=" * 50)
    print("模型解释性分析（SHAP）")
    print("=" * 50)

    import shap
    import math

    # ===== 数据准备 =====
    X_train_df = pd.DataFrame(X_train, columns=selected_features)
    X_test_df = pd.DataFrame(X_test, columns=selected_features)

    # ===== 计算 SHAP 值 =====
    rf_explainer = shap.TreeExplainer(best_rf)
    rf_shap_values = rf_explainer.shap_values(X_test_df)
    if isinstance(rf_shap_values, list):
        rf_shap_values = rf_shap_values[1]  # 取死亡（正类）

    lr_explainer = shap.LinearExplainer(best_lr, X_train_df)
    lr_shap_values = lr_explainer.shap_values(X_test_df)

    # ======================================================
    # 1️⃣ SHAP 特征重要性条形图（上下结构）
    # ======================================================
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    shap.summary_plot(
     rf_shap_values,
     X_test_df,
        plot_type="bar",
    show=False
    )
    plt.title("随机森林 SHAP 特征重要性")

    plt.subplot(2, 1, 2)
    shap.summary_plot(
    lr_shap_values,
    X_test_df,
    plot_type="bar",
    show=False
    )
    plt.title("逻辑回归 SHAP 特征重要性")

    plt.tight_layout()
    plt.savefig("shap_bar_rf_lr.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ======================================================
    # 2️⃣ SHAP 依赖图（6 个特征合成一张图）
    # ======================================================
    dependence_features = [
        "lab_5235_max",
        "lab_5237_min",
        "lab_5227_min",
        "lab_5225_range",
        "lab_5257_min",
        "age_month"
    ]
    dependence_features = [f for f in dependence_features if f in selected_features]

    # ---------- 随机森林 ----------
    plt.figure(figsize=(15, 10))
    plt.suptitle("随机森林 SHAP 依赖图", fontsize=16)

    for i, feat in enumerate(dependence_features, 1):
        ax = plt.subplot(2, 3, i)
        shap.dependence_plot(
            feat,
            rf_shap_values,
            X_test_df,
            interaction_index=None,
            show=False,
            ax=ax
        )
        ax.set_title(f"RF: {feat}", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("shap_dependence_rf.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ 随机森林依赖图已保存：shap_dependence_rf.png")

    # ---------- 逻辑回归 ----------
    plt.figure(figsize=(15, 10))
    plt.suptitle("逻辑回归 SHAP 依赖图", fontsize=16)

    for i, feat in enumerate(dependence_features, 1):
        ax = plt.subplot(2, 3, i)
        shap.dependence_plot(
            feat,
            lr_shap_values,
            X_test_df,
            interaction_index=None,
            show=False,
            ax=ax
    )
        ax.set_title(f"LR: {feat}", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("shap_dependence_lr.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ 逻辑回归依赖图已保存：shap_dependence_lr.png")



# ===================== 主函数 =====================
def main():
    # 1. 数据读取
    data = load_and_overview("data.xlsx")
    
    # 2. 数据探索
    exploratory_analysis(data)
    
    # 3. 数据预处理
    X_train, X_test, y_train, y_test, selected_features, scaler, data_clean = preprocess_data(data)
    
    # 4. 统计分析
    statistical_analysis(data_clean, selected_features)
    
    # 5. 无监督学习分析
    cluster_labels, silhouette_avg = unsupervised_analysis(data)
    
    # 6. 模型构建与调优（含SVM）
    best_lr, best_rf, best_svc = train_and_tune_models(X_train, y_train)
    
    # 7. 模型评估与可视化（含SVM）
    evaluate_and_visualize(best_lr, best_rf, best_svc, X_test, y_test, selected_features)
    
    # 8. 模型解释性分析（SHAP）
    shap_explanation(best_lr, best_rf, X_train, X_test, selected_features)
    
    # 结果清单
    print("\n" + "="*50)
    print("所有分析完成！生成文件清单：")
    print("1. 数据探索：descriptive_stats_before_imputation.csv、feature_histograms_before_imputation.png、grouped_boxplots_by_outcome.png")
    print("2. 无监督学习：kmeans_elbow_plot.png、cluster_outcome_ratio.png")
    print("3. 模型评估：roc_curve_three_models.png、confusion_matrix.png、feature_importance.png、model_results_three_models.csv")
    print("4. SHAP解释：")
    print("   - shap_bar_rf_lr.png（随机森林 & 逻辑回归 SHAP 条形图）")
    print("   - shap_dependence_rf.png（随机森林 6 个特征依赖图）")
    print("   - shap_dependence_lr.png（逻辑回归 6 个特征依赖图）")
    print("5. 统计分析：correlation_heatmap.png")
    print("="*50)

# 定义异常值处理函数（供外部调用）
def handle_outliers(df, features):
    df_clean = df.copy()
    for col in features:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
    return df_clean

# 运行主函数
if __name__ == "__main__":
    main()
