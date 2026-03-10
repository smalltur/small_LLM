#正则表达式库
import re
from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
with open("分词语料1.txt", "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        f.write(dataset[i]["text"] + "\n")
#进行分词操作

#打开分词语料文件
#将其储存到Word_List中
with open("./分词语料1.txt", "r", encoding = "utf-8") as Divided_File_1:
    Word_List = []
    Divided_File_1_Line = Divided_File_1.readline()
    while Divided_File_1_Line != "":
        #去除标点符号
        pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9]+'                                     #匹配非中文、非英文、非数字的字符,"+"表示如果有多个标点符号连续出现，也只替换为一个'</w>'
        Divided_File_1_Line = re.sub(pattern, '</w>', Divided_File_1_Line)          #将匹配到的字符替换为'</w>'
        Word_List.extend(Divided_File_1_Line.split('</w>'))
        ##
        # 以“我喜欢，你/n”为例，
        # 经过正则表达式处理后变为“我喜欢</w>你”， 
        # 再以'</w>'为分隔符进行分割，
        # 换行会保留为“”
        # 输出格式为：['我喜欢'，'你','']  
        ##
        Divided_File_1_Line = Divided_File_1.readline()

#关闭文件
#创建分词后储存分词数据的字典
chars_dict = {}

#词语最大长度为8
Word_Max_Length = 8

#开始进行分词
for i in range(1, Word_Max_Length + 1):
    for words in Word_List:
        if words != "" :
            if len(words) <= i:                     #当词语长度小于等于i时，直接将其作为一个词进行统计
                if words in chars_dict:
                    chars_dict[words] += 1
                else:
                    chars_dict[words] = 1
            else:
                for j in range(len(words) - i + 1):
                    word = words[j:j+i]
                    if word in chars_dict:
                        chars_dict[word] += 1
                    else:
                        chars_dict[word] = 1

#对字典降序排序
chars_dict = dict(sorted(chars_dict.items(), key=lambda item: item[1], reverse=True))
#输出分词结果
for word, count in chars_dict.items():  
    print(f"{word}: {count}")

