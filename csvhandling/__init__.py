import csv, sys
import numpy as np

class CsvHandling:
    def __init__(self):
        
        # Opened file that is being processed.
        self.file = None
        
        # Path to processed file.
        self.path = None
        
    def if_opened(self,path):
        try:
          self.file
        except NameError:
          return False
        else:
          if path == self.path:
              return True
          else:
              return False

    def open_file(self,path):
        if not self.if_opened(path):
            self.path = path
            self.file = open(self.path)
        return self.file
    
    def read_csv(self,file_path, target=None, has_header = True):
        with open(file_path, 'r') as f:
            num_cols = len(f.readline().split(","))
        data = np.loadtxt (file_path, delimiter=",", skiprows=1, dtype='float', usecols = range(0,num_cols-1))
        return data

    def write_csv(self,file_path, data):
        with open(file_path) as f:
            for line in data: f.write(",".join(line) + "\n")

    def get_target_index(self,file_path, target):
        try:
            with open(file_path) as f:
                r = csv.reader(f)
                line=r.__next__()
                return line.index(target)
        except ValueError:
            print("File " + file_path + " has no attribute " + target)
            sys.exit(1)
        
    def get_target(self,file_path, target):
        data = np.loadtxt (file_path, delimiter=",", skiprows=1, dtype=float, usecols = (self.get_target_index(file_path, target),))
        return data

    def get_features(self,file_path):
        with open(file_path) as f:
            r = csv.reader(f)
            line=r.__next__()
            return line