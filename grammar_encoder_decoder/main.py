def loop_line(lines, *arg):
    choice = None
    data = []

    import re
    for line in lines:
        line = re.sub('\n', '', line) # all lines have a trailing \n

        # don't ask for input
        if line[0] == "!":
            line = line[1:]
            if len(arg) == 2 and arg[0] == "2":
                line = line.split("\t")
                if arg[1] == None: line.insert(1, "None")
                else: line.insert(1, arg[1])
                line = " ".join(line)
            print line
            continue

        # ask for choice input
        if line[1] != "-":
            choice = raw_input(line)
            continue

        # ask for data input of corresponding choice
        line = line.split("-")
        if int(line[0]) == int(choice):
            data.append(raw_input(line[1]))

    return choice, data

def main():
    choice = "N"
    data = None
    path = "ui"

    file_format = ".txt"
    file_num = "1"

    from os.path import join
    while True:
        with open(join(path, file_num + file_format)) as f:
            choice, _ = loop_line(f.readlines())
        if choice == "0": break

        sub_choice = None
        hyperparameters = {}
        model_name = None
        while True:
            print "\n"
            with open(join(path, file_num + "_" + choice + file_format)) as f:
                sub_choice, data = loop_line(f.readlines(), choice, model_name)
                if sub_choice == "1" and choice == "2": model_name = "Custom"
            if sub_choice == "0": break
            f = open(join(path, file_num + "_" + choice + "c" + sub_choice + file_format))
            exec(f)
            f.close()

        print "\n\n\n"

main()
