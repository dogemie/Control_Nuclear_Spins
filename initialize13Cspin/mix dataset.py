import pandas as pd

csv1 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/ffd20230224_181117.csv')
csv2 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/ffd20230224_182557.csv')

csv3 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/ffd20230224_182713.csv')
csv4 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/ffd20230224_183322.csv')

csv5 = pd.concat([csv1, csv2])
csv6 = pd.concat([csv3, csv4])

csv7 = pd.concat([csv5, csv6])


csv8 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/ffd20230302_171625.csv')

csv9 = pd.concat([csv7, csv8])

csv10 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/ffd20230302_173714.csv')

csv11 = pd.concat([csv9, csv10])

print(csv11)
csv11.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/fixed_testdata.csv', index=False)