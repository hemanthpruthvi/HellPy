# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import seedir as sd
import copy as cp
from configobj import ConfigObj 
from termcolor import colored
from matplotlib import pyplot as plt
# from process_generator import *

class data_set:
    """
    Initialize object with the name of the top directory or config file
    """
    def __init__(self, name):
        self.meta = {}
        if (os.path.isdir(name)): 
            self.load_files(name)
        elif (os.path.isfile(name)): 
            self.load_from_txt(name)
        else: 
            print('Invalid input selection!') 
        self.load_settings()

    """
    Load files by reading config text file
    """
    def load_from_txt(self, name='meta.ini'):
        config = ConfigObj(name)
        self.meta['dir'] = config['dir']
        self.meta['dirtree'] = self.meta['dir'].split(os.sep)
        self.meta['topdir'] = self.meta['dirtree'][0]
    """
    Load files by entering directory names
    """
    def load_files(self, name):
        self.meta['topdir'] = name
        self.meta['dirtree'] = []
        self.list_dir(self.meta['topdir'])
        name = input('Select the data mode sub-directory: ')
        self.list_dir(name)
        name = input('Select the date for sub-directory: ')
        self.list_dir(name)
        name = input('Select the time for sub-directory: ')
        self.list_dir(name)
        self.meta['dir'] = os.sep.join(self.meta['dirtree'])
    """
    Load settings from ini file
    """
    def load_settings(self):
        settings_file = [self.meta['dir']+os.sep+file for file 
                         in sorted(os.listdir(self.meta['dir'])) if '.ini' in file]
        if (len(settings_file)==1):
            settings_file = settings_file[0]
            print('Settings loaded from ', settings_file)
        else: 
            print('No valid unique settings file found!') 
            return
        self.settings = ConfigObj(settings_file)
        self.meta['nlines']=0
        for i in range(32):
            if ('Line_'+str(i+1) in self.settings.sections):
                self.meta['nlines'] = i+1
            else:
                break
        print('Number of spectral lines in the data: ', self.meta['nlines'])
        self.meta['lines']=[]
        temp_dirs = [d for d in sorted(os.listdir(self.meta['dir']))
                                           if os.path.isdir(self.meta['dir']+os.sep+d)]
        print(temp_dirs)
        for i in range(self.meta['nlines']):
            self.meta[i] = {}
            self.meta[i]['roi'] = []
            self.meta[i]['files'] = []
            self.meta[i]['binning'] = []
            self.meta[i]['nframes'] = []
            linestr = 'Line_' + str(i+1)
            temp = self.settings[linestr]['Label']
            self.meta[i]['line'] = temp
            self.meta['lines'].append(temp)
            self.meta[i]['files'] = [self.meta['dir']+os.sep+temp+os.sep+file for 
                                     file in sorted(os.listdir(self.meta['dir']+os.sep+temp))]
            try:
                self.meta[i]['roi'].append(self.settings[linestr]['Camera_0\\ROI'])
                self.meta[i]['roi'].append(self.settings[linestr]['Camera_1\\ROI'])
                self.meta[i]['roi'].append(self.settings[linestr]['Camera_2\\ROI'])
                self.meta[i]['binning'].append(self.settings[linestr]['Camera_0\\Binning'])
                self.meta[i]['binning'].append(self.settings[linestr]['Camera_1\\Binning'])
                self.meta[i]['binning'].append(self.settings[linestr]['Camera_2\\Binning'])
            except:
                self.meta[i]['roi'].append(self.settings[linestr]['Camera_1\\ROI'])
                self.meta[i]['roi'].append(self.settings[linestr]['Camera_2\\ROI'])
                self.meta[i]['roi'].append(self.settings[linestr]['Camera_3\\ROI'])
                self.meta[i]['binning'].append(self.settings[linestr]['Camera_1\\Binning'])
                self.meta[i]['binning'].append(self.settings[linestr]['Camera_2\\Binning'])
                self.meta[i]['binning'].append(self.settings[linestr]['Camera_3\\Binning'])
            
            #
            self.meta[i]['nframes'].append(int(len(self.meta[i]['files'])/3))
            self.meta[i]['nframes'].append(3)          
            tempw = int(self.settings[linestr]['NWavePoints'])
            self.meta[i]['nframes'].append(tempw)
            temp1 = self.settings[linestr]['Filter']
            temp2 = self.settings[linestr]['Polarimeter\\Modulation']
            temp3 = 'Polarimeter\\' + temp2 + '\\NModulations'
            tempm = int(self.settings[temp1][temp3])
            self.meta[i]['nframes'].append(tempm)
            tempa = int(self.settings[linestr]['Polarimeter\\NAccumulations'])
            self.meta[i]['nframes'].append(tempa)
            #
            self.meta[i]['files'] = np.reshape(self.meta[i]['files'], 
                                               (3,self.meta[i]['nframes'][0]))
        try:
            self.read_data()
        except:
            print('Error reading a sample data!')
            
    """
    Read data by indices of line, camera and data 
    """
    def read_data(self, iline=0, icam=0, idata=0):
        temp = self.meta[iline]['files'][icam][idata]
        # print('Reading file: ', temp)
        self.data = np.fromfile(temp, dtype=np.int16, sep='')
        n = len(self.data)
        roi = int(self.meta[iline]['roi'][icam])
        self.data = np.reshape(self.data, [roi, roi, int(n/roi**2)], order='F')
        self.filename = temp
        
    """
    Display series of data as video
    """
    def show_data(self, pause=0.1):
        disp = plt.imshow(self.data[:,:,0])
        for i in range(self.data.shape[2]):
            disp.set_data(self.data[:,:,i])
            plt.pause(pause)
            plt.draw()
    """
    Save config as text
    """
    def save_as_txt(self, name='meta.ini'):
        config = ConfigObj(name)
        config['dir'] = self.meta['dir']
        config.write()
    """
    List out the details of the directory
    """
    def list_dir(self, name):
        self.meta['dirtree'].append(name)
        dirs = sorted(os.listdir(os.sep.join(self.meta['dirtree'])))
        print('List of sub-directories and files: ')
        for tempdir in dirs: 
            temp = os.sep.join(self.meta['dirtree']) + os.sep + tempdir
            if(os.path.isdir(temp)): 
                file_count, file_size = self.get_dir_details(temp)
                print(colored('%s \t %s \t %i'%(tempdir, file_size, file_count), 'cyan'))
        for tempdir in dirs:
            temp = os.sep.join(self.meta['dirtree']) + os.sep + tempdir
            if(os.path.isfile(temp)): 
                file_count, file_size = self.get_file_details(temp)
                print(colored('%s \t %s'%(tempdir, file_size), 'magenta'))
    """
    Get size and number of files for the given directory
    """
    def get_dir_details(self, name):
        file_count = 0
        file_size = 0
        KB = 1024.0
        MB = 1024*KB
        GB = 1024*MB
        TB = 1024*GB
        for temppath, tempdirs, tempfiles in os.walk(name):
            file_count += len(tempfiles)
            file_size += sum(os.path.getsize(os.path.join(temppath, name)) for name in tempfiles)
        if (file_size>=TB): 
            all_size = file_size/TB
            all_size = int(10*all_size)/10.0
            all_size = str(all_size) + ' TB'
        elif (file_size<TB and file_size>=GB):
            all_size = file_size/GB
            all_size = int(10*all_size)/10.0
            all_size = str(all_size) + ' GB'
        elif (file_size<GB and file_size>=MB):
            all_size = file_size/MB
            all_size = int(10*all_size)/10.0
            all_size = str(all_size) + ' MB'
        elif (file_size<MB and file_size>=KB):
            all_size = file_size/KB
            all_size = int(10*all_size)/10.0
            all_size = str(all_size) + ' KB'
        else:
            all_size = str(file_size) + ' B'
        return file_count, all_size
    """
    Get size for the file
    """
    def get_file_details(self, name):
        file_count = 1
        file_size = 0
        KB = 1024.0
        MB = 1024*KB
        GB = 1024*MB
        TB = 1024*GB
        file_size = os.path.getsize(name)
        if (file_size>=TB): 
            all_size = file_size/TB
            all_size = int(10*all_size)/10.0
            all_size = str(all_size) + ' TB'
        elif (file_size<TB and file_size>=GB):
            all_size = file_size/GB
            all_size = int(10*all_size)/10.0
            all_size = str(all_size) + ' GB'
        elif (file_size<GB and file_size>=MB):
            all_size = file_size/MB
            all_size = int(10*all_size)/10.0
            all_size = str(all_size) + ' MB'
        elif (file_size<MB and file_size>=KB):
            all_size = file_size/KB
            all_size = int(10*all_size)/10.0
            all_size = str(all_size) + ' KB'
        else:
            all_size = str(file_size) + ' B'
        return file_count, all_size
   
    
class data_cube:
    def __init__(self, directory, line, icam=0, icyc=0):
        self.directory = directory
        # self.iline = iline
        self.line = line
        self.icam = icam
        self.icyc = icyc
        # Load appropriate settings file
        settings_file = [directory+os.sep+file for file 
                         in sorted(os.listdir(directory)) if '.ini' in file]
        if (len(settings_file)==1):
            settings_file = settings_file[0]
            # print('Settings loaded from ', settings_file)
        else: 
            print('No valid unique settings file found!') 
            return
        self.settings = ConfigObj(settings_file)
        # filter and line settings
        # line_str = 'Line_' + str(iline+1)
        # self.line = self.settings[line_str]['Label']
        for i in range(1,31):
            if(self.settings['Line_'+str(i)]['Label']==self.line):
                line_str = 'Line_' + str(i)
                self.iline = i
                break
            else:
                self.iline = 0
        if (self.iline==0):
            print('No suitable data found for the line!')
            return
        #
        try:
            cam_str = 'Camera_' + str(icam) 
            self.roi = int(self.settings[line_str][cam_str+'\\ROI'])
            self.binning = int(self.settings[line_str][cam_str+'\\Binning'])
        except:
            cam_str = 'Camera_' + str(icam+1) 
            self.roi = int(self.settings[line_str][cam_str+'\\ROI'])
            self.binning = int(self.settings[line_str][cam_str+'\\Binning'])
        files = [directory+os.sep+self.line+os.sep+file for 
                                     file in sorted(os.listdir(directory+os.sep+self.line))]
        nfiles = len(files)//3
        self.file = files[nfiles*icam+icyc]
        # print('Data loading from ', self.file)
        self.data = np.fromfile(self.file, dtype=np.int16, sep='')
        n = len(self.data)
        self.data = np.reshape(self.data, [self.roi, self.roi, int(n/self.roi**2)], 
                               order='F')
        
    def show_data(self, pause=0.1):
        disp = plt.imshow(self.data[:,:,0])
        for i in range(self.data.shape[2]):
            disp.set_data(self.data[:,:,i])
            plt.pause(pause)
            plt.draw()