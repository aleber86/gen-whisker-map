import argparse as aps


class File_reader:

    def __init__(self, file_name : str, headers = False,
                 headers_size = 1, comments = '#') -> None:
        self.file_name = file_name
        self.list_size = 0
        self.records = 0
        self.columns = 0
#        self.headers_data = None
        self.file_data_raw = None
#        self.headers = headers
#        self.headers_size = headers_size
        self.comment_wild_card = comments

    def read_file(self) -> None:
        """Function reads file. Gets data and collect attribs like column span and lenght

        Args:
            None

        Returns:
            None

        """
        self.file_data_raw = []
        with open(self.file_name, "r") as file_to_read:
            status = True
            while status:
                line = file_to_read.readline().split()
                if line != []:
                    if self.__comment_finder(line):  self.file_data_raw.append(line)
                else:
                    status = False
#        if self.headers:
#            self.headers_data = self.file_data_raw[:self.headers_size]
#            self.file_data_raw = self.file_data_raw[self.headers_size:]

        self.__config_object()

    def __comment_finder(self, argument : list):
        to_find = argument[0]
        status = True
        for letter in to_find:
            if letter == ' ':
                continue
            elif letter not in (' ', '#'):
                break
            elif letter == '#':
                status = False
                break
        return status

    def __config_object(self) -> None:
        """Function config attribs like column span and lenght of data

        Args:
            None

        Returns:
            None
        """
        self.records = len(self.file_data_raw)
        self.columns = len(self.file_data_raw[0])

    def get_data(self) -> str:
        """Function returns file data without headers

        Args:
            None

        Returns:
            self.file_data_raw : str (file data)

        """
        return self.file_data_raw

    def get_full_atribs(self) -> dict:
        attribs = {"records" : self.records, "columns" : self.columns}
        return attribs

class Cmd_parser:
    """Command line argument parser class"""

    def __init__(self, program : str = '', menu_args : dict = {})->None:
        """Init class funtion

        Args:
            program : str (program name)
            menu_args : dict (arguments scheme)

        Returns:
            None
        """
        self.menu = aps.ArgumentParser(prog = program)
        self.__argument_config(args = menu_args)

    def __argument_config(self, args : dict) -> None:
        """Sets the command lines arguments from a dictionary object.

        Args:
            args : dict

        Returns:
            None

        """


        for key, value in args.items():
            self.menu.add_argument(key, value['long_string'],
                                   action=value['action'],
                                   help=value['help'],
                                   default=value['default'],
                                   nargs=value['nargs'],
                                   type = value['type'])


    def get_menu(self) -> aps.Namespace:
        """Gets the command line arguments namespace

        Args:
            None

        Returns:
            Namespace

        """
        return self.menu.parse_args()




if __name__ == '__main__':

    """
    cmd_line_args = {'-f' : {'long_string' : '--file-config',
                             'action' : 'store',
                             'help' : 'Parse a config file for the program.',
                             'type' : str,
                             'nargs' : '?',
                             'default' : ''},
                     '-i' : {'long_string' : '--input-file',
                             'action' : 'store',
                             'help' : 'Sets the auxiliary file to calculate ...',
                             'nargs' : '?',
                             'type' : str,
                             'default' : ''},
                     '-o' : {'long_string' : '--output-file',
                             'action' : 'store',
                             'help' : 'Sets the output file of the calculation.',
                             'nargs' : '?',
                             'type' : str,
                             'default' : ''},
                     '-r' : {'long_string' : '--raster-dimension',
                             'action' : 'store',
                             'help' : 'Sets the raster dimension size',
                             'type': int,
                             'nargs' : 2,
                             'default' : [4096,2048]}
                    }
    args = Cmd_parser('')
    menu = args.get_menu()
    """


    file_read = File_reader("test.dat")
    file_read.read_file()
    data = file_read.get_data()

    print(data)
