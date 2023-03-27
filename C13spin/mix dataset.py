import pandas as pd
import os
from datetime import datetime as dt                         # 시간을 출력하기 위한 라이브러리 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

date = dt.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
print(date)


folder1 = os.listdir('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/data_set/atested')

folder2 = os.listdir('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/data_set/compare')


csv_fin1 = pd.DataFrame()
csv_fin2 = pd.DataFrame()

for files in folder1:
    csv1 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/data_set/atested/'+files)
    csv_fin1 = pd.concat([csv_fin1, csv1], axis=0)
    
for files in folder2:
    csv2 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/data_set/compare/'+files)
    csv_fin2 = pd.concat([csv_fin2, csv2], axis=0)

# print(csv_fin1)
csv_fin1.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/test_data/atesteddata_' + printdate + '.csv', index=False)
csv_fin1.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/testdata.csv', index=False)

csv_fin2.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/test_data/comparedata_' + printdate + '.csv', index=False)
csv_fin2.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/test_compare_data.csv', index=False)
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin1['Al']
# Y = csv_fin1['Ap']
# Z = csv_fin1['cost']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('cost')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_cost_figure1.png')

# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# X2 = csv_fin2['Al']
# Y2 = csv_fin2['Ap']
# Z2 = csv_fin2['cost']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('cost')
# # plt.zlabel('cost')
# ax2.plot_trisurf(X2, Y2, Z2, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_cost_figure2.png')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin1['Al']
# Y = csv_fin1['Ap']
# Z = csv_fin1['Xtau'] * csv_fin1['XN'] * 2 + csv_fin1['Ztau'] * csv_fin1['ZN']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('time')
# # plt.zlabel('time')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_time_figure1.png')

# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# X2 = csv_fin2['Al']
# Y2 = csv_fin2['Ap']
# Z2 = csv_fin2['Xtau'] * csv_fin2['XN'] * 2 + csv_fin2['Ztau'] * csv_fin2['ZN']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('time')
# # plt.zlabel('time')
# ax2.plot_trisurf(X2, Y2, Z2, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_time_figure2.png')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin1['Al']
# Y = csv_fin1['Ap']
# Z = csv_fin1['Xtau']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('Xtau')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_Xtau1_figure.png')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin1['Al']
# Y = csv_fin1['Ap']
# Z = csv_fin1['XN']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('XN')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_XN1_figure.png')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin1['Al']
# Y = csv_fin1['Ap']
# Z = csv_fin1['Ztau']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('Ztau')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_Ztau1_figure.png')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin1['Al']
# Y = csv_fin1['Ap']
# Z = csv_fin1['ZN']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('ZN')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_ZN1_figure.png')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin2['Al']
# Y = csv_fin2['Ap']
# Z = csv_fin2['Xtau']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('Xtau')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_Xtau2_figure.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = csv_fin2['Al']
Y = csv_fin2['Ap']
Z = csv_fin2['XN']
plt.xlabel('Al')
plt.ylabel('Ap')
plt.title('XN')
# plt.zlabel('cost')
ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_XN2_figure.png')


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin2['Al']
# Y = csv_fin2['Ap']
# Z = csv_fin2['Ztau']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('Ztau')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_Ztau2_figure.png')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin2['Al']
# Y = csv_fin2['Ap']
# Z = csv_fin2['ZN']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('ZN')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_ZN2_figure.png')



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin1['Al']
# Y = csv_fin1['Ap']
# Z = csv_fin1['XN'] * csv_fin1['Xtau']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('XN * Xtau')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_Xtime1_figure.png')


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin1['Al']
# Y = csv_fin1['Ap']
# Z = csv_fin1['ZN'] * csv_fin1['Ztau']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('ZN * Ztau')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_Ztime1_figure.png')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin2['Al']
# Y = csv_fin2['Ap']
# Z = csv_fin2['XN'] * csv_fin2['Xtau']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('XN * Xtau')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_Xtime2_figure.png')


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = csv_fin2['Al']
# Y = csv_fin2['Ap']
# Z = csv_fin2['ZN'] * csv_fin2['Ztau']
# plt.xlabel('Al')
# plt.ylabel('Ap')
# plt.title('ZN * Ztau')
# # plt.zlabel('cost')
# ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
# # plt.show()
# plt.savefig('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/figure/' + printdate + '_Ztime2_figure.png')