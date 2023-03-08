import pandas as pd

csv1 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_104751.csv')

csv2 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_105241.csv')

csv3 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_105321.csv')

csv4 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_105415.csv')

csv5 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_105532.csv')

csv6 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_105650.csv')

csv6_2 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_105706.csv')

csv7 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_132046.csv')

csv8 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_132349.csv')

csv9 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_132511.csv')

csv10 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_132910.csv')

csv11 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_133118.csv')

csv12 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_133611.csv')

csv13 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_133738.csv')

csv14 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_152610.csv')

csv15 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_152906.csv')

csv16 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_153110.csv')

csv17 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_153228.csv')

csv18 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_154611.csv')

csv19 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/newdata/ffd20230308_154801.csv')


csvfin = pd.concat([csv1, csv2, csv3, csv4, csv5, csv6, csv6_2, csv7, csv8, csv9, csv10, csv11, csv12, csv13, csv14, csv15, csv16, csv17, csv18, csv19], ignore_index=True)

print(csvfin)
csvfin.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/Init13C_Sec/testdata.csv', index=False)