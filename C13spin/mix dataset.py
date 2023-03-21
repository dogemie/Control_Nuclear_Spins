import pandas as pd
import os
from datetime import datetime as dt                         # 시간을 출력하기 위한 라이브러리 

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

csv_fin2.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/test_data/comparedata_' + printdate + '.csv', index=False)