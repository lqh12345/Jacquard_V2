import json
import os
import re

# Defining the path of the root directory
root_dir = "./result_J_0_finish"

# Defining the path of the data files
data_file_path = os.path.join(root_dir, "jacquard0_11_batchsize_32_J_0_10-labelbee.json")

# Defining the path of the false_data file
false_data_file_path = os.path.join(root_dir, "false_data.txt")

# Defining the path of the false_path file
false_path_file_path = os.path.join(root_dir, "false_path.txt")

tn_count = 0
fn_count = 0
del_count = 0
num=0

if __name__ == '__main__':
    with open(data_file_path) as f:
        data = json.load(f)

    for item in data:
        print('num:',num)
        num += 1
        filename = item["fileName"]
        match = re.search(r'\d+', filename)
        ans = int(match.group())
        print('ans:', ans, end=" ")

        result = json.loads(item["result"])
        try:
            class_1 = result["step_1"]["result"][0]["result"]["class-1"]
        except KeyError:
            print("There is a blank value")
            continue

        if class_1 == "TN(1)":
            print("TN")
            tn_count += 1
            continue

        if class_1 == "FN(1)":
            print("FN")
            fn_count += 1
            with open(false_data_file_path) as f_data, open(false_path_file_path) as f_path:
                data_line = f_data.readlines()[ans].strip()
                path_line = f_path.readlines()[ans].strip()
                print(path_line)
                with open(path_line, "r") as f_txt:
                    old_data = f_txt.read().strip()
                    print("original data is:")
                    print(old_data)
                with open(path_line, "w") as f_txt:
                    f_txt.write(data_line + "\n" + old_data)
                with open(path_line, "r") as f_txt:
                    new_data = f_txt.read().strip()
                    print("new data is:")
                    print(new_data)

        elif class_1 == "DEL(1)":
            print("DEL")
            del_count += 1
            with open(false_path_file_path) as f_path:
                path_line = f_path.readlines()[ans].strip()
                print(path_line)
                suffix_list = ['_grasps.txt', '_mask.png', '_perfect_depth.tiff', '_RGB.png', '_stereo_depth.tiff']
                dir_path, file_name = os.path.split(path_line)
                name_without_suffix = file_name[:-11]
                for suffix in suffix_list:
                    full_file_name = name_without_suffix + suffix
                    new_path = os.path.join(dir_path, full_file_name)
                    try:
                        os.remove(new_path)
                        print(full_file_name + " was deleted successfully!")
                    except FileNotFoundError:
                        print("Can not find "+ full_file_name)

        else:
            continue

    total_count = tn_count + fn_count + del_count
    tn_ratio = tn_count / total_count
    fn_ratio = fn_count / total_count
    del_ratio = del_count / total_count
    print("TN count:", tn_count, "({:.2%})".format(tn_ratio))
    print("FN count:", fn_count, "({:.2%})".format(fn_ratio))
    print("DEL count:", del_count, "({:.2%})".format(del_ratio))
