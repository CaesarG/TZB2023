import os
import pandas as pd


def get_rcs_ifft(annotation_line):
    line = annotation_line.split()
    path_rcs = line[0]
    ground_truth = int(line[1])
    return path_rcs, ground_truth


with open('dataRCS/annotations/test.txt', 'r') as f:
    annotation_lines = f.readlines()
length = len(annotation_lines)
if not os.path.exists('validDATA'):
    os.mkdir('validDATA')
GT = []


def pd_toExcel(data, fileName='GT.xlsx'):
    dfData = {
        'GroundTruth': data,
    }
    df = pd.DataFrame(dfData)
    writer = pd.ExcelWriter(fileName, engine='xlsxwriter')
    df.style.set_properties(**{'text-align': 'center', 'border': '1px solid black'}).to_excel(writer,
                                                                                              sheet_name='Sheet1',
                                                                                              index=False)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format_border = workbook.add_format({'border': 1})
    worksheet.conditional_format('A1:XFD1048576', {'type': 'no_blanks', 'format': format_border})
    # 保存 Excel 文件
    writer.close()


for i in range(length):
    rcs, gt = get_rcs_ifft(annotation_lines[i])
    rcs = rcs.replace('/', '\\')
    GT.append(gt)
    os.system('copy {} validDATA\\frame_{}.mat'.format(rcs, i + 1))
    ...

pd_toExcel(GT)
