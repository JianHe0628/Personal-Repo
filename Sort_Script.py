import os
import shutil
import argparse

def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default=os.getcwd())
    parser.add_argument('--h',type=int,default=1)
    return parser.parse_args()
class Sort:
    def __init__(self):
        self.type = ['Word','PDFs','PowerPoint','Excel','Pictures','TextFiles','Code','Videos','Audio']
        self.path = _parse().path
    
    def main(self):
        self.scan_files()
        if _parse().h == 1:
            for x in self.arglist:
                self.scan_header(x)
            self.scan_files()
        for x in range(0,8):
            self.file_sorter(self.arglist[x],self.type[x]) #Sort Word, PDFs, PowerPoint, Pictures, TextFiles, Code
        self.unsorted_folder(self.unsorted)
        print('Files Reorganized. ')

    def scan_files(self):
        self.Excels,self.Words,self.Pictures,self.PDFs,self.Code,self.Text,self.unsorted,self.ppt,self.vid,self.audio = [],[],[],[],[],[],[],[],[],[]
        file = os.listdir(self.path)
        print(f'Scanning files in path {file}')
        for y in file:
            x = y.lower()
            if x.endswith('.csv') or x.endswith('.xlsx') or x.endswith('.xlsm') or x.endswith('.xls'):
                self.Excels.append(x)
            elif x.endswith('docx') or x.endswith('.doc'):
                self.Words.append(x)
            elif x.endswith('.jpeg') or x.endswith('.jpg') or x.endswith('.png') or x.endswith('.tiff') or x.endswith('.tif'):
                self.Pictures.append(x)
            elif x.endswith('pdf'):
                self.PDFs.append(x)
            elif x.endswith('.py') or x.endswith('.c'):
                self.Code.append(x)
            elif x.endswith('.txt'):
                self.Text.append(x)
            elif x.endswith('.pptx') or x.endswith('.ppt'):
                self.ppt.append(x)
            elif x.endswith('.mp4') or x.endswith('.mov') or x.endswith('.avi'):
                self.vid.append(x)
            elif x.endswith('.mp3') or x.endswith('.wav') or x.endswith('.flac'):
                self.audio.append(x)
            else:
                self.unsorted.append(x)
        self.arglist = [self.Words,self.PDFs,self.ppt,self.Excels,self.Pictures,self.Text,self.Code,self.vid,self.audio]
    
    def file_sorter(self,list,type):
        for x in list:
            try:
                if not os.path.isdir(x):
                    self.mkdirectory_handler(type,x,x)
                else:
                    print(f'{x} is a folder. Does not need sorting.')
            except Exception as e:
                print(f'{x} not sorted. Error: {e}')

    def scan_header(self,list):
        for x in list:
            try:
                if '~' in x:
                    splitted_names = x.split('~')
                    newx = x.replace(f'{splitted_names[0]}~','')
                    self.mkdirectory_handler(splitted_names[0],x,newx)
                else:
                    pass
            except Exception as e:
                print(f'{x} not sorted. Error: {e}')

    def unsorted_folder(self,list):
        for x in list:
            try:
                if not os.path.isdir(x):
                    self.mkdirectory_handler('Unsorted_Items',x,x)
                else:
                    print(f'{x} is a folder. Does not need sorting.')
            except Exception as e:
                print(f'{x} not sorted. Error: {e}')

    def mkdirectory_handler(self,header,ori,new):
        newpath = f'{self.path}\\{header}'
        if os.path.exists(newpath):
            print(f'{header} folder already exist.')
        else:
            os.mkdir(header)
            print(f'New Folder [{header}] created.') 
        shutil.move(ori,f'{newpath}\\{new}')

if __name__ == '__main__':
    sort_obj = Sort()
    sort_obj.main()
        
