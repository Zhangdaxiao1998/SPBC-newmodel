import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
import gzip
import pickle


# # 检查并下载 Git LFS 文件
# if not os.path.exists("rsf_model3.pkl"):
#     os.system("git lfs pull")  # 确保 Git LFS 下载模型

# 定义一个函数用于预测和展示结果
# ========== 预测新患者的生存曲线 ==========
def predict_survival(new_patient, target_time):
    """ 使用已训练好的 RSF 模型预测新患者的生存曲线，并计算置信区间 """

    with gzip.open("rsf_final1.pkl.gz", "rb") as f:
        rsf, X_train, train_yt_merge_y = pickle.load(f)

    # 计算新患者的生存函数
    surv_fn = rsf.predict_survival_function(new_patient)

    # 获取时间点和对应的生存概率
    time_points = surv_fn[0].x  # 提取时间点
    survival_probs = surv_fn[0].y  # 提取对应的生存概率

    # ========== 计算置信区间 ==========
    n_bootstrap = 2  # 你可以调大，比如 50 以提高稳定性
    survival_curves = []

    for _ in range(n_bootstrap):
        X_resampled, y_resampled = resample(X_train, train_yt_merge_y, random_state=_)
        rsf_boot = rsf
        rsf_boot.fit(X_resampled, y_resampled)
        surv_fn_boot = rsf_boot.predict_survival_function(new_patient)
        survival_prob_interp = np.interp(time_points, surv_fn_boot[0].x, surv_fn_boot[0].y)  # 统一时间点
        survival_curves.append(survival_prob_interp)

    survival_curves = np.array(survival_curves)  # 现在所有曲线长度一致

    # 计算均值和置信区间
    mean_survival = survival_curves.mean(axis=0)
    # mean_survival = survival_probs.mean(axis=0)
    se_survival = survival_curves.std(axis=0) / np.sqrt(n_bootstrap)
    ci_lower = mean_survival - 1.96 * se_survival
    ci_upper = mean_survival + 1.96 * se_survival

    fig, ax = plt.subplots()
    ax.plot(time_points, mean_survival, label="Survival Probability", color='blue')
    # ax.fill_between(time_points, ci_lower, ci_upper, color='blue', alpha=0.2, label="95% CI")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Survival Curve")
    ax.legend()


    survival_prob_at_t = np.interp(target_time, time_points, mean_survival)
    death_risk_at_t = 1 - survival_prob_at_t
    # 查找对应时间点的置信区间
    # lower_ci_at_t = np.interp(target_time, time_points, ci_lower)
    # upper_ci_at_t = np.interp(target_time, time_points, ci_upper)

    # 显示预测结果
    st.write(f"新患者在 {target_time} 个月后的死亡风险: {death_risk_at_t:.4f}")
    st.write(f"新患者在 {target_time} 个月后的生存概率: {survival_prob_at_t:.4f}")
    # st.write(f"95% 置信区间: ({lower_ci_at_t:.4f}, {upper_ci_at_t:.4f})")
    st.pyplot(fig)


# Streamlit 页面设置
st.title("女性第二原发乳腺癌患者生存预测模型")
st.header("请输入新患者的信息:")

# 输入表单
# 年龄
age_options = {1: "20~45岁", 2: "46~68岁", 3: "大于68岁"}
age = st.selectbox('年龄(岁)', options=list(age_options.keys()), format_func=lambda x: age_options[x])

latency = st.slider('两次肿瘤的确诊间隔时间(月)', min_value=2, max_value=300, value=50, step=1)
Tumor_size = st.slider('SPBC肿瘤大小（mm）', min_value=0, max_value=200, value=10, step=1)

# nodes
nodes_examined_options = {1: "≤2", 2: "3~6", 3: ">6"}
nodes_examined = st.selectbox('区域淋巴结检测数（个）', options=list(nodes_examined_options.keys()),
                              format_func=lambda x: nodes_examined_options[x])

# first_site
first_site_options = {1: "乳腺", 2: "女性生殖系统", 3: "消化系统", 4: "皮肤癌",
                      5: "内分泌系统", 6: "泌尿系统", 7: "呼吸系统", 8: "淋巴癌", 9: "其他位点"}
first_site = st.selectbox('第一原发恶性肿瘤位点', options=list(first_site_options.keys()),
                          format_func=lambda x: first_site_options[x])

# marital
marital_options = {1: "已婚", 2: "丧偶", 3: "单身", 4: "离异"}
marital = st.selectbox('婚姻状况', options=list(marital_options.keys()),
                       format_func=lambda x: marital_options[x])

# histology_type
histology_type_options = {1: "8500/3: 浸润性导管癌",
                          2: "8520/3: 小叶癌",
                          3: "8522/3: 导管/小叶混合型癌",
                          4: "其它组织学类型"}
histology_type = st.selectbox('SPBC组织学类型', options=list(histology_type_options.keys()),
                              format_func=lambda x: histology_type_options[x])

# LN_Axillary(I-II)
LN_Axillary_options = {0: "阴性", 1: "阳性"}
LN_Axillary = st.selectbox("腋窝I/II级淋巴结检测", options=list(LN_Axillary_options.keys()),
                           format_func=lambda x: LN_Axillary_options[x])


# Stage
stage_options = {1: "Stage I期", 2: "Stage II期", 3: "Stage III期", 4: "Stage IV期"}
stage = st.selectbox('SPBC临床综合分期', options=list(stage_options.keys()), format_func=lambda x: stage_options[x])

# grade
grade_options = {1: "Grade I: Well differentiated",
                 2: "Grade II: Moderately differentiated",
                 3: "Grade III: Poorly differentiated",
                 4: "Grade IV: Undifferentiated (Anaplastic)"}
grade = st.selectbox('SPBC肿瘤分级', options=list(grade_options.keys()), format_func=lambda x: grade_options[x])

# Subtype
Subtype_options = {1: "Luminal-A型",
                   2: "Luminal-B型",
                   3: "HER2阳性型",
                   4: "三阴性型"}
Subtype = st.selectbox('SPBC亚型', options=list(Subtype_options.keys()), format_func=lambda x: Subtype_options[x])

# surgery_1
surgery_1st_options = {0: "未接受手术", 1: "接受手术"}
surgery_1st = st.selectbox('第一原发肿瘤手术情况', options=list(surgery_1st_options.keys()),
                           format_func=lambda x: surgery_1st_options[x])
# radiotherapy_1
radiotherapy_1st_options = {0: "未接受放疗", 1: "接受放疗"}
radiotherapy_1st = st.selectbox('第一原发肿瘤放疗情况', options=list(radiotherapy_1st_options.keys()),
                                format_func=lambda x: radiotherapy_1st_options[x])

# chemotherapy_1st
chemotherapy_1st_options = {0: "未接受化疗", 1: "接受化疗"}
chemotherapy_1st = st.selectbox('第一原发肿瘤化疗情况', options=list(chemotherapy_1st_options.keys()),
                                format_func=lambda x: chemotherapy_1st_options[x])

# surgery_2nd
surgery_2nd_options = {0: "未接受手术", 1: "接受手术"}
surgery_2nd = st.selectbox('SPBC手术情况', options=list(surgery_2nd_options.keys()),
                           format_func=lambda x: surgery_2nd_options[x])

# radiotherapy_2nd
radiotherapy_2nd_options = {0: "未接受放疗", 1: "接受放疗"}
radiotherapy_2nd = st.selectbox('SPBC放疗情况', options=list(radiotherapy_2nd_options.keys()),
                                format_func=lambda x: radiotherapy_2nd_options[x])

# chemotherapy_2nd
chemotherapy_2nd_options = {0: "未接受化疗", 1: "接受化疗"}
chemotherapy_2nd = st.selectbox('SPBC化疗情况', options=list(chemotherapy_2nd_options.keys()),
                                format_func=lambda x: chemotherapy_2nd_options[x])

# Surg_Rad_Seq
Surg_Rad_Seq_options = {0: "仅手术/放疗",
                        1: "术后放疗",
                        2: "术前/中放疗"}
Surg_Rad_Seq = st.selectbox('SPBC手术/化疗次序', options=list(Surg_Rad_Seq_options.keys()),
                            format_func=lambda x: Surg_Rad_Seq_options[x])

# systemic_therapy
systemic_therapy_options = {0: "未接受全身治疗", 1: "接受全身治疗"}
systemic_therapy = st.selectbox('SPBC全身治疗情况', options=list(systemic_therapy_options.keys()),
                                format_func=lambda x: systemic_therapy_options[x])

# Neoadjuvant_Therapy
Neoadjuvant_Therapy_options = {0: "未接受治疗",
                               1: "部分反应",
                               2: "完全反应",
                               3: "无反应"}
Neoadjuvant_Therapy = st.selectbox('SPBC全身治疗情况', options=list(Neoadjuvant_Therapy_options.keys()),
                                   format_func=lambda x: Neoadjuvant_Therapy_options[x])


# 输入目标时间
target_time = st.slider('请输入您想要预测的时间点(月)', min_value=2, max_value=280, value=60, step=1)

new_patient = pd.DataFrame({"Age": [age],
                            "Marital": [marital],
                            "First_site": [first_site],
                            "Latency": [latency],
                            "Histology": [histology_type],
                            "LN_Axillary(I-II)": [LN_Axillary],
                            "Stage": [stage],
                            "Grade": [grade],
                            "nodes_examined": [nodes_examined],
                            "Tumor_size": [Tumor_size],
                            "Subtype": [Subtype],
                            "Surgery_1st": [surgery_1st],
                            "Radio_1st": [radiotherapy_1st],
                            "Chemo_1st": [chemotherapy_1st],
                            "Surgery_2nd": [surgery_2nd],
                            "Radio_2nd": [radiotherapy_2nd],
                            "Chemo_2nd": [chemotherapy_2nd],
                            "Systemic_therapy": [systemic_therapy],
                            "Neoadjuvant_Therapy": [Neoadjuvant_Therapy],
                            "Surg_Rad_Seq": [Surg_Rad_Seq]})


# 按钮触发预测
if st.button('输出模型预测结果'):
    predict_survival(new_patient, target_time)
