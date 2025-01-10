import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, pearsonr


def remove_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data >= lower_bound) & (data <= upper_bound)


files_path = "./visualization/vis_result/morph/totalall_femur"
morph_name = ["NSA",
              "FHR",
              "FO",
              "FHC",
              "FDA",
              "FNA"]
info_lists = []
dice_points = []
show_points = []
for file in os.listdir(files_path):
    with open(os.path.join(files_path, file), 'r', encoding='utf-8') as f:
        morph_dic = json.load(f)

    info_list = []
    for key, value in morph_dic.items():
        if len(value) == 0:
            continue
        info = []
        for k, v in value.items():
            for k_, v_ in v.items():
                info.append(v_)
            info_list.append(np.array(info).reshape([-1, 7]).T)
    info_lists.append(np.concatenate(info_list, -1))

for i in range(6):
    dice_points = []
    show_points = []
    line_x = []
    line_y = []
    for info in info_lists:
        x_mask = remove_outliers_iqr(info[0])
        y_mask = remove_outliers_iqr(info[i + 1])
        mask = x_mask & y_mask
        x_filtered = info[0][mask]
        y_filtered = info[i + 1][mask]

        dice_points.append(np.mean(x_filtered))
        show_points.append(np.mean(y_filtered))
        line_x.append(x_filtered)
        line_y.append(y_filtered)

    line_x = np.concatenate(line_x)
    line_y = np.concatenate(line_y)

    # PLOT
    slope, intercept, r_value, p_value, std_err = linregress(line_x, line_y)
    regression_line = slope * line_x + intercept
    rho, _ = pearsonr(line_x, line_y)  # 计算皮尔逊相关系数

    plt.figure(figsize=(2.5, 2.5))
    plt.rcParams['text.antialiased'] = True
    # Model
    scatter = plt.scatter(dice_points, show_points, c=np.linspace(0, 1, len(dice_points)), s=100, cmap='viridis', alpha=0.8)
    # scatter = plt.scatter(line_x, line_y, c=np.linspace(0, 1, len(line_x)), s=100, cmap='viridis', edgecolor='k', alpha=0.8)
    # Line
    plt.plot(line_x, regression_line, color='blue', linewidth=2)
    sns.regplot(x=line_x, y=line_y, scatter=False, ci=95, line_kws={"color": "blue", "linewidth": 1})

    # 4. 设置标题和标签
    plt.tick_params(axis='both', direction='in', length=2, width=1, colors='black', labelsize=10)
    plt.title(r"$R^2={:.2f}, \rho={:.2f}$".format(r_value ** 2, rho), fontsize=10)
    plt.xlabel('DSC ↑', fontsize=10, labelpad=0)
    plt.ylabel(rf"${morph_name[i]}$" + " ↓", fontsize=10, labelpad=0)
    if i in [0, 4, 5]:
        plt.ylabel(rf"${morph_name[i]}$" + "° ↓", fontsize=10, labelpad=0)

    plt.tight_layout()
    plt.xlim(min(dice_points) - 0.025, max(dice_points) + 0.025)
    plt.ylim(min(show_points) - 2.5, max(show_points) + 2.5)
    if i in [1, 2, 3]:
        plt.ylim(min(show_points) - 0.5, max(show_points) + 0.5)
    # plt.show()
    plt.savefig(os.path.join(files_path, str(i)+morph_name[i]+".svg"), format="svg")
