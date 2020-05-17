import os

path=r'E:\tsl_file\python_project\all_datas\captcha_datas'
path2=r'E:\tsl_file\python_project\all_datas'
import re
file_name=os.listdir(path)
for each_name in file_name:
    # print(each_name)
    abs_path=os.path.join(path,each_name)
    notation=re.split('\_|\.',each_name)[1]
    with open(os.path.join(path2,'annotation_train.txt'),'a+') as f:
        f.write('{} {}\n'.format('captcha_datas/{}'.format(each_name),notation))
    # print(abs_path,notation)