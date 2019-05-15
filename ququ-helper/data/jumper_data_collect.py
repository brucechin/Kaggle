# coding: utf-8
import pandas as pd
columns = ['Rptdt','Fenddt','AnanmID','Ananm','ReportID','InstitutionID','Brokern','Feps','Fpe','Fturnover','Stkcd']

def get_target_reportID(filename):
    current_employment = {}
    jump_record = []
    df = pd.read_csv(filename,encoding='gbk')
    data = df[['AnanmID','ReportID','InstitutionID']]
    #data.sort_values("AnanmID",inplace=True)
    count = 0
    not_jump_count = 0
    null_count = 0
    for index, row in data.iterrows():
        #print(row)
        if(row['AnanmID'] == -1):
            null_count += 1
            continue

        if(row['AnanmID'] in current_employment.keys()):
            #print(current_employment[row['AnanmID']],row['InstitutionID'])
            if(current_employment[row['AnanmID']][0] == row['InstitutionID']):
                not_jump_count += 1
                current_employment[row['AnanmID']][1] = row['ReportID']  #update his newest reportID in current insititution
            else:#indicate that he change his insititution（跳槽）
                count += 1
                jump_record.append(current_employment[row['AnanmID']][1])
                jump_record.append(row['ReportID'])
                current_employment[row['AnanmID']][1] = row['ReportID']  #again we need to update his reportID anyway
        else:
            append_item = [row['InstitutionID'],row['ReportID']]
            current_employment[row['AnanmID']] = append_item
    #print(jump_record)
    print("jump count is : {}".format(count))
    print("not jump count : {}".format(not_jump_count))
    print("people null count : {}".format(null_count))
    print(len(current_employment))
    return jump_record

def generate_jump_stats(filename,jump_reportID_list):
    df = pd.read_csv(filename, encoding='gbk')
    print(len(jump_reportID_list))
    jump_data = df[df['ReportID'].isin(jump_reportID_list)]
    #print(jump_data)
    jump_data.to_csv('jump-record'+filename,encoding='gbk')


filename = "csmar-2001-2008-splited.csv"
jump_record = get_target_reportID(filename)
generate_jump_stats(filename,jump_record)