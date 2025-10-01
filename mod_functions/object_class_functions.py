class file_reader:
    def __init__(self, file_name : str, headers = False, headers_size = 1) -> None:
        self.file_name = file_name
        self.list_size = 0
        self.records = 0
        self.columns = 0
        self.headers_data = None
        self.file_data_raw = None
        self.headers = headers
        self.headers_size = headers_size

    def read_file(self) -> None:
        """Function reads file. Gets data and collect attribs like column span and lenght"""
        self.file_data_raw = []
        with open(self.file_name, "r") as file_to_read:
            status = True
            while status:
                line = file_to_read.readline().split()
                if line != []:
                    self.file_data_raw.append(line)
                else:
                    status = False
        if self.headers:
            self.headers_data = self.file_data_raw[:self.headers_size]
            self.file_data_raw = self.file_data_raw[self.headers_size:]

        self.__config_object()

    def __config_object(self) -> None:
        """Function config attribs like column span and lenght of data"""
        self.records = len(self.file_data_raw)
        self.columns = len(self.file_data_raw[0])

    def get_data(self) -> str:
        """Function returns file data without headers"""
        return self.file_data_raw

    def get_full_atribs(self) -> dict:
        attribs = {"records" : self.records, "columns" : self.columns}
        return attribs




if __name__ == '__main__':
    read = file_reader("../resultado_64_el_wm.dat")
    read.read_file()
    datos = read.get_data()
    att = read.get_full_atribs()
    print(att)
#    print(datos)


