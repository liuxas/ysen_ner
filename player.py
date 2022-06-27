from sys import flags
from numpy.core.fromnumeric import size
import pandas as pd
from ipdb import set_trace
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# print(matplotlib.matplotlib_fname())
# font = FontProperties(fname=r"/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
# plt.title("标题", fontproperties=font)
# plt.xlabel("x轴标签", fontproperties=font)
# plt.ylabel("y轴标签", fontproperties=font)
# plt.savefig("vsion/result_"+".png")
# plt.show()

path = ["vison/result_bvas_only.csv"]
mask = 0
for file in path:
    flag = file.split("_")[-1].split(".")[0]
    data = pd.read_csv(file)
    data = data.sort_values(by="value",ascending=False)
    data = data.iloc[0:10]
    # data = data.set_index("keywords")
    if mask==0:
        plt.title('120份病历关键词柱状图')
        plt.bar(data.symptom, data.value)
        plt.xticks(data.symptom, data.symptom, rotation=25,size=6)
        for a, b in zip(data.symptom, data.value):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=17)
        plt.savefig("vison/result_"+flag+".png")
        mask = 1
    else:
        plt.figure(figsize=(6,6))
        label = data["symptom"]
        values = data["value"]
        plt.pie(values,labels=label,autopct='%1.1f%%')
        plt.title('120份病历关键词饼图')
        plt.savefig("vison/result_"+flag+"b_"+".png")
set_trace()
    