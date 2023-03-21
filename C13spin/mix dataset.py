import pandas as pd
import os
from datetime import datetime as dt                         # 시간을 출력하기 위한 라이브러리 

date = dt.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
print(date)


folder = os.listdir('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/data_set/atested')

csv_fin = pd.DataFrame()

for files in folder:
    csv = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/data_set/atested/'+files)
    csv_fin = pd.concat([csv_fin, csv], axis=0)

print(csv_fin)
csv_fin.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/C13spin/test_data/testdata_' + printdate + '.csv', index=False)
