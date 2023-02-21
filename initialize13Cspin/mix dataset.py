import pandas as pd

csv1 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_131848.csv')
csv2 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_131918.csv')

csv3 = pd.concat([csv1, csv2], axis=0)

csv4 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_131949.csv')

csv5 = pd.concat([csv3, csv4], axis=0)

csv6 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_132004.csv')

csv7 = pd.concat([csv5, csv6], axis=0)

csv8 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_143129.csv')

csv9 = pd.concat([csv7, csv8], axis=0)

csv10 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_143131.csv')

csv11 = pd.concat([csv9, csv10], axis=0)

csv12 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_143345.csv')

csv13 = pd.concat([csv11, csv12], axis=0)

csv14 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_143446.csv')

csv15 = pd.concat([csv13, csv14], axis=0)

csv16 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_153415.csv')

csv17 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_153420.csv')

csv18 = pd.concat([csv16, csv17], axis=0)

csv19 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_153429.csv')

csv20 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_153437.csv')

csv21 = pd.concat([csv19, csv20], axis=0)

csv22 = pd.concat([csv18, csv21], axis=0)

csv23 = pd.concat([csv15, csv22], axis=0)

csv24 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_165252.csv')

csv25 = pd.concat([csv23, csv24], axis=0)

csv26 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_165307.csv')

csv27 = pd.concat([csv25, csv26], axis=0)

csv28 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_165345.csv')

csv29 = pd.concat([csv27, csv28], axis=0)

csv30 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/newdata/dataset20230221_165402.csv')

csv31 = pd.concat([csv29, csv30], axis=0)

print(csv31)
csv31.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/testdata_second.csv', index=False)