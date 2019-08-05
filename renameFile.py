import os


def rename_files(dir_path, number):
    """
    批量重命名文件
    参数：
        dirPath：文件路径
        number：之前图片的总和
    """
    file_list = os.listdir(dir_path)

    index = 0
    for item in file_list:
        oldname = dir_path + r"\\" + file_list[index]
        newname = dir_path + r"\\" + str(number + index) + ".jpg"
        os.rename(oldname, newname)

        index += 1

if __name__ == "__main__":
    imageDir = 'img_test'
    rename_files(imageDir, 0)

