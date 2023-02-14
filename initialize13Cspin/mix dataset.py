import pandas as pd

csv1 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230213_171227.csv')
csv2 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230214_124716.csv')

csv3 = csv1.drop(['tau'], axis=1)

csv3.rename(columns={'6': 'fun'}, inplace=True)

csv4 = pd.concat([csv3, csv2], axis=0)



csv5 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230214_161003.csv')


csv6 = pd.concat([csv4, csv5], axis=0)

csv7 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230214_164155.csv')

csv8 = pd.concat([csv6, csv7], axis=0) #2345

csv9 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230214_165240.csv')

csv10 = pd.concat([csv8, csv9], axis=0)  #2383

csv11 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230214_170520.csv')

csv12 = pd.concat([csv10, csv11], axis=0)  #2416

csv13 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230214_171743.csv')

csv14 = pd.concat([csv12, csv13], axis=0)  #2452

csv15 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230214_172800.csv')

csv16 = pd.concat([csv14, csv15], axis=0)  

csv17 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230214_172912.csv')

csv18 = pd.concat([csv16, csv17], axis=0)  

csv19 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230214_173111.csv')

csv20 = pd.concat([csv18, csv19], axis=0)  #2562

print(csv20)
csv20.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/testdata.csv', index=False)