
import time
import csv
label_csv = open("../result/testCSV.csv", 'w', encoding='utf-8', newline='')
label_csv_writer = csv.writer(label_csv)

label_csv_writer.writerow(["data", "test label", "predict label"])

for i in range (10):
    x = i
    y = i + 1
    nowtime = time.localtime()
    nowtime_style = time.strftime("%Y-%m-%d %H:%M:%S", nowtime)
    label_csv_writer.writerow([nowtime_style, str(x), str(y)])


label_csv.close()