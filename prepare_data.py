import os

import numpy as np

ADHD = np.load("./data/ADHD_label.npy").T
ADNI = np.load("./data/ADNI_label.npy").T
AD = np.load("./data/AD_label.npy").T
ASD = np.load("./data/ASD_label.npy").T
dHCP = np.load("./data/dHCP_label.npy").T
HCP = np.load("./data/HCP_label.npy").T
CHD = np.load("./data/CHD_label.npy")
ABIDE2 = np.load("./data/ABIDE2_label.npy").T
CCNP = np.load("./data/CCNP_label.npy").T
CCNP_PEK = np.load("./data/CCNP_PEK_label.npy").T
print(ADHD.shape)
CHD[:,0] = CHD[:,0] / 12
# print(CHD[:,0])
# print(np.unique(ADHD[:,1]))
# print(np.unique(ADNI[:,1]))
# print(np.unique(ASD[:,1]))
# print(np.unique(dHCP[:,1]))
# print(np.unique(HCP[:,1]))
# print(np.unique(CHD[:,1]))
# print(np.unique(ABIDE2[:,1]))
# print(np.unique(CCNP[:,1]))
# print(np.unique(CCNP_PEK[:,1]))
#
# print(np.unique(ADHD[:,2]))
# print(np.unique(ADNI[:,2]))
# print(np.unique(ASD[:,2]))
# print(np.unique(dHCP[:,2]))
# print(np.unique(HCP[:,2]))
# print(np.unique(CHD[:,2]))
# print(np.unique(ABIDE2[:,2]))
# print(np.unique(CCNP[:,2]))
# print(np.unique(CCNP_PEK[:,2]))

# ADHD = np.concatenate([ADHD[:,:,:40962],ADHD[:,:,163842:163842+40962]],axis=2)
# ADNI = np.concatenate([ADNI[:,:,:40962],ADNI[:,:,163842:163842+40962]],axis=2)
# AD = np.concatenate([AD[:,:,:40962],AD[:,:,163842:163842+40962]],axis=2)
# ASD = np.concatenate([ASD[:,:,:40962],ASD[:,:,163842:163842+40962]],axis=2)
# dHCP = np.concatenate([dHCP[:,:,:40962],dHCP[:,:,163842:163842+40962]],axis=2)
# HCP = np.concatenate([HCP[:,:,:40962],HCP[:,:,163842:163842+40962]],axis=2)
# CHD = np.concatenate([CHD[:,:,:40962],CHD[:,:,163842:163842+40962]],axis=2)
# ABIDE2 = np.concatenate([ABIDE2[:,:,:40962],ABIDE2[:,:,163842:163842+40962]],axis=2)
# CCNP = np.concatenate([CCNP[:,:,:40962],CCNP[:,:,163842:163842+40962]],axis=2)
# CCNP_PEK = np.concatenate([CCNP_PEK[:,:,:40962],CCNP_PEK[:,:,163842:163842+40962]],axis=2)

# No_CHD_label = np.vstack([ADNI[:,:3], ASD[:,:3], dHCP[:,:3], HCP[:,:3]], CHD[:,:3], ABIDE2[:,:3], CCNP[:,:3],CCNP_PEK[:,:3])
# label = np.concatenate([["ADNI"]*ADNI.shape[0],["ASD"]*ASD.shape[0],["dHCP"]*dHCP.shape[0],["HCP"]*HCP.shape[0], ["CHD"]*CHD.shape[0],["ASD"]*ABIDE2.shape[0],["CCNP"]*CCNP.shape[0],["CCNP"]*CCNP_PEK.shape[0]]).reshape([-1,1])
# No_CHD_label = np.hstack([No_CHD_label,label])
# np.save("./data_surfclip/No_CHDaADHD_label_text.npy", No_CHD_label)

#
No_CHD_label = np.vstack([ADHD[:,:3], ADNI[:,:3], AD[:,:3], ASD[:,:3], dHCP[:,:3], HCP[:,:3], ABIDE2[:,:3], CCNP[:,:3],CCNP_PEK[:,:3]])
label = np.concatenate([["ADHD"]*ADHD.shape[0],["ADNI"]*ADNI.shape[0],["ADNI"]*AD.shape[0],["ASD"]*ASD.shape[0],["dHCP"]*dHCP.shape[0],["HCP"]*HCP.shape[0],["ASD"]*ABIDE2.shape[0],["CCNP"]*CCNP.shape[0],["CCNP"]*CCNP_PEK.shape[0]]).reshape([-1,1])
No_CHD_label = np.hstack([No_CHD_label,label])
np.save("./data_surfclip/No_CHD_label_text.npy", No_CHD_label)
#
No_HCP_label = np.vstack([ADHD[:,:3], ADNI[:,:3], AD[:,:3], ASD[:,:3], dHCP[:,:3], CHD[:,:3], ABIDE2[:,:3], CCNP[:,:3],CCNP_PEK[:,:3]])
label = np.concatenate([["ADHD"]*ADHD.shape[0],["ADNI"]*ADNI.shape[0],["ADNI"]*AD.shape[0],["ASD"]*ASD.shape[0],["dHCP"]*dHCP.shape[0],["CHD"]*CHD.shape[0],["ASD"]*ABIDE2.shape[0],["CCNP"]*CCNP.shape[0],["CCNP"]*CCNP_PEK.shape[0]]).reshape([-1,1])
No_HCP_label = np.hstack([No_HCP_label,label])
np.save("./data_surfclip/No_HCP_label_text.npy", No_HCP_label)

# No_dHCP_label = np.vstack([ADHD[:,:3], ADNI[:,:3], AD[:,:3], ASD[:,:3], HCP[:,:3], CHD[:,:3], ABIDE2[:,:3], CCNP[:,:3],CCNP_PEK[:,:3]])
# label = np.concatenate([["ADHD"]*ADHD.shape[0],["ADNI"]*ADNI.shape[0],["ADNI"]*AD.shape[0],["ASD"]*ASD.shape[0],["HCP"]*HCP.shape[0],["CHD"]*CHD.shape[0],["ASD"]*ABIDE2.shape[0],["CCNP"]*CCNP.shape[0],["CCNP"]*CCNP_PEK.shape[0]]).reshape([-1,1])
# No_dHCP_label = np.hstack([No_dHCP_label,label])
# np.save("./data_surfclip/No_dHCP_label_text.npy", No_dHCP_label)

# No_ASD_label = np.vstack([ADHD[:,:3], ADNI[:,:3], AD[:,:3] dHCP[:,:3], HCP[:,:3], CHD[:,:3], ABIDE2[:,:3], CCNP[:,:3],CCNP_PEK[:,:3]])
# label = np.concatenate([["ADHD"]*ADHD.shape[0],["ADNI"]*ADNI.shape[0],["ADNI"]*AD.shape[0],["dHCP"]*dHCP.shape[0],["HCP"]*HCP.shape[0],["CHD"]*CHD.shape[0],["ASD"]*ABIDE2.shape[0],["CCNP"]*CCNP.shape[0],["CCNP"]*CCNP_PEK.shape[0]]).reshape([-1,1])
# No_ASD_label = np.hstack([No_ASD_label,label])
# np.save("./data_surfclip/No_ASD_label_text.npy", No_ASD_label)

No_ADNI_label = np.vstack([ADHD[:,:3], ADNI[:,:3], ASD[:,:3], dHCP[:,:3], HCP[:,:3], CHD[:,:3], ABIDE2[:,:3], CCNP[:,:3],CCNP_PEK[:,:3]])
label = np.concatenate([["ADHD"]*ADHD.shape[0],["ADNI"]*ADNI.shape[0],["ASD"]*ASD.shape[0],["dHCP"]*dHCP.shape[0],["HCP"]*HCP.shape[0],["CHD"]*CHD.shape[0],["ASD"]*ABIDE2.shape[0],["CCNP"]*CCNP.shape[0],["CCNP"]*CCNP_PEK.shape[0]]).reshape([-1,1])
No_ADNI_label = np.hstack([No_ADNI_label,label])
np.save("./data_surfclip/No_ADNI_label_text.npy", No_ADNI_label)
#
No_ADHD_label = np.vstack([ADNI[:,:3], AD[:,:3], ASD[:,:3], dHCP[:,:3], HCP[:,:3], CHD[:,:3], ABIDE2[:,:3], CCNP[:,:3],CCNP_PEK[:,:3]])
label = np.concatenate([["ADNI"]*ADNI.shape[0],["ADNI"]*AD.shape[0],["ASD"]*ASD.shape[0],["dHCP"]*dHCP.shape[0],["HCP"]*HCP.shape[0],["CHD"]*CHD.shape[0],["ASD"]*ABIDE2.shape[0],["CCNP"]*CCNP.shape[0],["CCNP"]*CCNP_PEK.shape[0]]).reshape([-1,1])
No_ADHD_label = np.hstack([No_ADHD_label,label])
np.save("./data_surfclip/No_ADHD_label_text.npy", No_ADHD_label)

# No_ABIDE2_label = np.vstack([ADHD[:,:3], ADNI[:,:3], ASD[:,:3], dHCP[:,:3], HCP[:,:3], CHD[:,:3], CCNP[:,:3],CCNP_PEK[:,:3]])
# label = np.concatenate([["ADHD"]*ADHD.shape[0],["ADNI"]*ADNI.shape[0],["ASD"]*ASD.shape[0],["dHCP"]*dHCP.shape[0],["HCP"]*HCP.shape[0],["CHD"]*CHD.shape[0]]).reshape([-1,1])
# No_ABIDE2_label = np.hstack([No_ABIDE2_label,label])
# np.save("./data_surfclip/No_ABIDE2_label_text.npy", No_ABIDE2_label)



# No_CHD = np.concat([ADNI, ASD, dHCP, HCP],axis=0)
# np.save("./data_surfclip/No_CHDaADHD.npy", No_CHD)
#
# No_ADHD_mean = np.mean(No_CHD,axis=0)
# No_ADHD_mean = np.mean(No_ADHD_mean,axis=1)
# No_ADHD_std = np.std(No_CHD,axis=0)
# No_ADHD_std = np.std(No_ADHD_std,axis=1)
# print(No_ADHD_mean.shape)
# print(No_ADHD_std.shape)
# np.save("./data_surfclip/No_CHDaADHD_mean.npy", No_ADHD_mean)
# np.save("./data_surfclip/No_CHDaADHD_std.npy", No_ADHD_std)

# No_CHD_label = np.concat([ADHD, ADNI, AD, ASD, dHCP, HCP, ABIDE2, CCNP, CCNP_PEK],axis=0)
# np.save("./data_surfclip/No_CHD.npy", No_CHD_label)
# No_CHD_mean = np.mean(No_CHD_label.astype("float32"),axis=0)
# No_CHD_mean = np.mean(No_CHD_mean,axis=1)
# No_CHD_std = np.std(No_CHD_label.astype("float32"),axis=0)
# No_CHD_std = np.std(No_CHD_std,axis=1)
# np.save("./data_surfclip/No_CHD_mean.npy", No_CHD_mean)
# np.save("./data_surfclip/No_CHD_std.npy", No_CHD_std)
#
# print(No_CHD_label[2999])
# print(np.where((np.sum(np.mean(No_CHD_label,axis=2)==0,axis=1)==6))[0])
#
# No_HCP_label = np.concat([ADHD, ADNI, AD, ASD, dHCP, CHD, ABIDE2, CCNP, CCNP_PEK],axis=0)
# np.save("./data_surfclip/No_HCP.npy", No_HCP_label)
# No_CHD_mean = np.mean(No_HCP_label.astype("float32"),axis=0)
# No_CHD_mean = np.mean(No_CHD_mean,axis=1)
# No_CHD_std = np.std(No_HCP_label.astype("float32"),axis=0)
# No_CHD_std = np.std(No_CHD_std,axis=1)
# np.save("./data_surfclip/No_HCP_mean.npy", No_CHD_mean)
# np.save("./data_surfclip/No_HCP_std.npy", No_CHD_std)

# No_dHCP_label = np.concat([ADHD, ADNI, ASD, HCP, CHD, ABIDE2, CCNP, CCNP_PEK],axis=0)
# np.save("./data_surfclip/No_dHCP.npy", No_dHCP_label)

# No_ASD_label = np.concat([ADHD, ADNI, dHCP, HCP, CHD, ABIDE2, CCNP, CCNP_PEK],axis=0)
# np.save("./data_surfclip/No_ASD.npy", No_ASD_label)

# No_ADNI_label = np.concat([ADHD, ADNI, ASD, dHCP, HCP, CHD, ABIDE2, CCNP, CCNP_PEK],axis=0)
# np.save("./data_surfclip/No_ADNI.npy", No_ADNI_label)
# No_CHD_mean = np.mean(No_ADNI_label.astype("float32"),axis=0)
# No_CHD_mean = np.mean(No_CHD_mean,axis=1)
# No_CHD_std = np.std(No_ADNI_label.astype("float32"),axis=0)
# No_CHD_std = np.std(No_CHD_std,axis=1)
# np.save("./data_surfclip/No_ADNI_mean.npy", No_CHD_mean)
# np.save("./data_surfclip/No_ADNI_std.npy", No_CHD_std)
#
# No_ADHD_label = np.concat([ADNI, AD, ASD, dHCP, HCP, CHD, ABIDE2, CCNP, CCNP_PEK],axis=0)
# np.save("./data_surfclip/No_ADHD.npy", No_ADHD_label)
# No_CHD_mean = np.mean(No_ADHD_label.astype("float32"),axis=0)
# No_CHD_mean = np.mean(No_CHD_mean,axis=1)
# No_CHD_std = np.std(No_ADHD_label.astype("float32"),axis=0)
# No_CHD_std = np.std(No_CHD_std,axis=1)
# np.save("./data_surfclip/No_ADHD_mean.npy", No_CHD_mean)
# np.save("./data_surfclip/No_ADHD_std.npy", No_CHD_std)

# No_ABIDE2_label = np.concat([ADHD, ADNI, ASD, dHCP, HCP, CHD, CCNP, CCNP_PEK],axis=0)
# np.save("./data_surfclip/No_ABIDE2.npy", No_ABIDE2_label)

# ================================== compute the mean and std===================
# No_ADHD = np.load("./data_surfclip/No_ABIDE2.npy")
# No_ADHD_mean = np.mean(No_ADHD,axis=0)
# No_ADHD_mean = np.mean(No_ADHD_mean,axis=1)
# No_ADHD_std = np.std(No_ADHD,axis=0)
# No_ADHD_std = np.std(No_ADHD_std,axis=1)
# print(No_ADHD_mean.shape)
# print(No_ADHD_std.shape)
# np.save("./data_surfclip/No_ABIDE2_mean.npy", No_ADHD_mean)
# np.save("./data_surfclip/No_ABIDE2_std.npy", No_ADHD_std)
#
# No_ADHD = np.load("./data_surfclip/No_HCP.npy")
# No_ADHD_mean = np.mean(No_ADHD,axis=0)
# No_ADHD_mean = np.mean(No_ADHD_mean,axis=1)
# No_ADHD_std = np.std(No_ADHD,axis=0)
# No_ADHD_std = np.std(No_ADHD_std,axis=1)
# print(No_ADHD_mean.shape)
# print(No_ADHD_std.shape)
# np.save("./data_surfclip/No_HCP_mean.npy", No_ADHD_mean)
# np.save("./data_surfclip/No_HCP_std.npy", No_ADHD_std)


#================================== prepare train and test====================
# ADHD = np.load("./data_surfclip/No_ADHD.npy")
# age = np.load("./data_surfclip/No_ADHD_label_text.npy")
# ADHD_label= np.load("./data_surfclip/No_ADHD_text_embeddings.npy")
# indices = np.arange(ADHD_label.shape[0])             # 生成 [0, 1, 2, ..., 9999]
# np.random.shuffle(indices)
# test_indices = indices[:500]           # 前500个作为测试集
# train_indices = indices[500:]
# test_data = ADHD[test_indices]         # 形状 (500, 6, 40962)
# train_data = ADHD[train_indices]       # 形状 (9500, 6, 40962)
# test_data_label = ADHD_label[test_indices]         # 形状 (500, 6, 40962)
# train_data_label = ADHD_label[train_indices]       # 形状 (9500, 6, 40962)
# np.save("./data_surfclip/No_ADHD_train.npy", train_data)
# np.save("./data_surfclip/No_ADHD_test.npy", test_data)
# np.save("./data_surfclip/No_ADHD_train_text_embeddings.npy", train_data_label)
# np.save("./data_surfclip/No_ADHD_test_text_embeddings.npy", test_data_label)


#=====================================test=====================================
# age = np.load("./data_surfclip/No_HCP_label_text.npy")[:,0].astype(float)
#
# # 2. 定义年龄段分组（示例分为5组，根据实际数据调整）
# bins = [0, 6, 18, 30, 40, 60,100]  # 年龄区间：0-12, 13-18, 19-30, 31-60, 61-100
# labels = ['0-6', '7-18', '19-30','31-40', '41-60', '61-100']
# print(sum((age>=7)&(age<18)))
# # 3. 将连续年龄离散化为年龄段
# age_groups = np.digitize(age, bins=bins) - 1  # 获取每个样本所属的组别
# print(age_groups)
# # 4. 统计原始分布
# unique_groups, original_counts = np.unique(age_groups, return_counts=True)
# print(original_counts)
# # 5. 计算分层抽样比例
# total_samples = 300
# proportions = original_counts / original_counts.sum()
# print(proportions)
# target_counts = np.round([0.16* total_samples,0.16* total_samples,0.16* total_samples,0.16* total_samples,0.16* total_samples,0.16* total_samples] ).astype(int)
# print(target_counts)
# # 6. 调整总数误差（确保总和=500）
# delta = total_samples - target_counts.sum()
# target_counts[np.argmax(target_counts)] += delta  # 将余数加到最大的组
#
# # 7. 分层随机抽样
# selected_indices = []
# final_counts = np.zeros_like(target_counts)
#
# for i, group in enumerate(unique_groups):
#     # 获取当前组所有样本的原始索引
#     group_indices = np.where(age_groups == group)[0]
#
#     # 检查样本是否足够
#     if len(group_indices) < target_counts[i]:
#         raise ValueError(f"Age group {labels[i]} has only {len(group_indices)} samples, but needs {target_counts[i]}")
#
#     # 无放回随机抽样
#     selected = np.random.choice(group_indices, size=target_counts[i], replace=False)
#     selected_indices.extend(selected)
#     final_counts[i] = len(selected)
#
# # 8. 打乱顺序（可选）
# indices = np.array(selected_indices)
# np.random.shuffle(indices)
#
# # 9. 输出结果
# print("分层抽样结果：")
# for i, (label, count) in enumerate(zip(labels, final_counts)):
#     print(f"{label}岁: {count}人")
#
# print("\n所有被选中的索引：")
# print(indices)
#
# mask = np.ones(len(age), dtype=bool)
# mask[indices] = False            # 标记索引位置为False
#
# ADHD = np.load("./data_surfclip/No_HCP.npy")
# age = np.load("./data_surfclip/No_HCP_label_text.npy")
# ADHD_label= np.load("./data_surfclip/No_HCP_text_embeddings.npy")
#
# test_data = ADHD[indices]         # 形状 (500, 6, 40962)
# train_data = ADHD[mask]       # 形状 (9500, 6, 40962)
# test_data_label = ADHD_label[indices]         # 形状 (500, 6, 40962)
# train_data_label = ADHD_label[mask]       # 形状 (9500, 6, 40962)
# print(test_data.shape)
# print(train_data.shape)
# print(test_data_label.shape)
# print(train_data_label.shape)
# np.save("./data_surfclip/No_HCP_train.npy", train_data)
# np.save("./data_surfclip/No_HCP_test.npy", test_data)
# np.save("./data_surfclip/No_HCP_train_text_embeddings.npy", train_data_label)
# np.save("./data_surfclip/No_HCP_test_text_embeddings.npy", test_data_label)
# np.save("./data_surfclip/No_HCP_indices.npy", indices)

