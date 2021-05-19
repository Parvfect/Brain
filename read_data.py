

def main(file_name):
    data_arr = []
    
    fileVal = open(file_name, "r")
    lines = fileVal.readlines()
    for line in lines:
        data_arr.append(float(line))

    return data_arr
