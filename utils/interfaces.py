import colorize  # noqa


class Tape():

    def __init__(self, tape_data, blank_char=5):

        self.pointer = 0
        self.tape_data = tape_data
        self.tape_length = len(self.tape_data)
        self.blank_char = blank_char

    def move_head(self, action):
        self.pointer += action  # actions in range [-1,0,1]

    def read_head(self):
        if self.pointer >= 0 and self.pointer < self.tape_length:
            return self.tape_data[self.pointer]
        else:
            return self.blank_char

    def write_head(self, content):
        if self.pointer >= 0 and self.pointer < self.tape_length:
            self.tape_data[self.pointer] = content
        else:
            pass

    def display(self):
        chr_str = ""
        for i in range(self.tape_length):
            tape_int = self.tape_data[i]
            tape_symbol = self.int_to_symbol(i)
            if i == self.pointer:
                tape_symbol = colorize.colorize(
                    tape_symbol, "green", bold=False, highlight=False)
            chr_str += tape_symbol
        return chr_str

    def int_to_symbol(self, val):

        symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        if val == self.blank_char:
            return " "
        else:
            return symbols[val]

    def query(self, address):
        if address >= 0 and address < self.tape_length:
            return self.tape_data[address]
        else:
            return self.blank_char


# class OutputTape(Tape):
#     def __init__(self, tape_data, blank_char=5):
#         super().__init__(length, length)


class Memory():
    pass
