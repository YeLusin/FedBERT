import argparse
import os
import re


def parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", type=str, help="file to be sampled")  #  切分的文件路径
    parser.add_argument("--output_dir", type=str)  # 切分后的文件保存路径
    args = parser.parse_args()

    return args


# filename = './wikitext-103-raw/wiki.test.raw'
# output_dir = './wikitext-103-raw/shuffle'

# 将文件按一级标题划分
def read_line(args):
    filename = args.input_filename
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    file = open(filename, "r", encoding="utf-8") 
    paragraph = []
    f_id, title_flag = 1, -1
    
    for line in file:
        tokens = line.strip().split(' ')        
        if tokens[0] == "=" and tokens[-1] == "=" and tokens[1] != "=":  # 判断是否是一级标题
            title_flag += 1
        
        if title_flag != 1:
            paragraph = open(output_dir+'/'+str(f_id)+'.raw', 'a')  # 以写模式"w" open会造成覆盖，应该以追加模式"a"
            paragraph.writelines(line)
            paragraph.close()

        # 遇到新的一级标题，存储在新的文件
        elif title_flag == 1:
            # print("Paragraph file saved: ", f_id)
            f_id += 1
            title_flag = 0 
            paragraph = open(output_dir+'/'+str(f_id)+'.raw', 'a')  
            paragraph.writelines(line)
            paragraph.close()      
    print("--------------------")
    print("Input: ", filename)
    print("Output: ", output_dir)
    print("Total processed: ", f_id, "files.")
    print("--------------------")
    file.close()

args = parameters()

def main():
    read_line(args)

if __name__ == "__main__":
    main()
