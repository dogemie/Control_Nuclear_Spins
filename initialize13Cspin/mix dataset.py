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

csv21 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_005451.csv')

csv22 = pd.concat([csv20, csv21], axis=0)  #2562

csv23 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_005455.csv')

csv24 = pd.concat([csv22, csv23], axis=0)  #2562

csv25 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_005505.csv')

csv26 = pd.concat([csv24, csv25], axis=0)  #3026

csv27 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_151411.csv')

csv28 = pd.concat([csv26, csv27], axis=0)  #3026

csv29 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_151451.csv')

csv30 = pd.concat([csv28, csv29], axis=0)  #3026

csv31 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_151709.csv')

csv32 = pd.concat([csv30, csv31], axis=0)  #3026

csv33 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_153100.csv')

csv34 = pd.concat([csv32, csv33], axis=0)  #3026

csv35 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_153409.csv')

csv36 = pd.concat([csv34, csv35], axis=0)  #3026

csv37 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_151305.csv')

csv38 = pd.concat([csv36, csv37], axis=0)  #7552

csv39 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_171255.csv')

csv40 = pd.concat([csv38, csv39], axis=0)  #7552

csv41 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_173216.csv')

csv42 = pd.concat([csv40, csv41], axis=0)  #7552

csv43 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230215_173254.csv')

csv44 = pd.concat([csv42, csv43], axis=0)  #7630

csv45 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230216_165136.csv')

csv46 = pd.concat([csv44, csv45], axis=0)  #7630

csv47 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230216_165224.csv')

csv48 = pd.concat([csv46, csv47], axis=0)  #7630

csv49 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230216_165230.csv')

csv50 = pd.concat([csv48, csv49], axis=0)  #7630

csv51 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230216_165500.csv')

csv52 = pd.concat([csv50, csv51], axis=0)  #7630

csv53  = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230216_165632.csv')

csv54 = pd.concat([csv52, csv53], axis=0)  #7630

csv55 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230216_165651.csv')

csv56 = pd.concat([csv54, csv55], axis=0)  #13322

csv57 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_101652.csv')

csv58 = pd.concat([csv56, csv57], axis=0)  #13322

csv59 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_101815.csv')

csv60 = pd.concat([csv58, csv59], axis=0)  #13322

csv61 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_101853.csv')

csv62 = pd.concat([csv60, csv61], axis=0)  #13322

csv63 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_102019.csv')

csv64 = pd.concat([csv62, csv63], axis=0)  #13322

csv65 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_102021.csv')

csv66 = pd.concat([csv64, csv65], axis=0)  #13322

csv67 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_102100.csv')

csv68 = pd.concat([csv66, csv67], axis=0)  #13322

csv69 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_152552.csv')

csv70 = pd.concat([csv68, csv69], axis=0)  #13322

csv71 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_152553.csv')

csv72 = pd.concat([csv70, csv71], axis=0)  #13322

csv73 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_152621.csv')

csv74 = pd.concat([csv72, csv73], axis=0)  #13322

csv75 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_152632.csv')

csv76 = pd.concat([csv74, csv75], axis=0)  #13322

csv77 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_152636.csv')

csv78 = pd.concat([csv76, csv77], axis=0)  #13322

csv79 = pd.read_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/dataset20230217_152835.csv')

csv80 = pd.concat([csv78, csv79], axis=0)  #17200

print(csv80)
csv80.to_csv('C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/initialize13Cspin/file/testdata.csv', index=False)