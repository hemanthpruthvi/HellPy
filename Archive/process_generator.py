# -*- coding: utf-8 -*-
"""
"""

import os
import termcolor
import configobj

"""
Create config file by selecting directories for various data
"""
def generate_config():
    config = configobj.ConfigObj('config.ini')
    top_dir = config['topdir']
    #
    termcolor.cprint('\nSelect the directory for science data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['science']['directory'] = direc
    config.write()
    #
    termcolor.cprint('\nSelect the directory for darks data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['darks']['directory'] = direc
    config.write()
    #
    termcolor.cprint('\nSelect the directory for flats data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['flats']['directory'] = direc
    config.write()
    #
    termcolor.cprint('\nSelect the directory for pcalibration data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['pcalibration']['directory'] = direc
    config.write()
    #
    termcolor.cprint('\nSelect the directory for targetplate data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['targetplate']['directory'] = direc
    config.write()
    #
    termcolor.cprint('\nSelect the directory for pinhole data: \n', 'grey', 'on_white')
    direc = select_dir(top_dir)
    config['pinhole']['directory'] = direc
    config.write()
    
"""
Select directory for needed data, given top level directory
"""
def select_dir(top_dir):
    if not os.path.isdir(top_dir):
        print(top_dir, ' is not a valid directory!')
        return
    dir_tree = top_dir.split(os.sep)
    path = os.sep.join(dir_tree)
    list_dir(path)
    sub_dir = input('Select the next level sub-directory in ' 
                    + path +  '\nType # and ENTER to go back a level \nPress ENTER to finalize selection : ')
    if (sub_dir == ''):
        final_dir = os.sep.join(dir_tree)
        if os.path.isdir(final_dir):
            print('Selected directory is: ', final_dir)
        else:
            print(final_dir, ' is not a valid directory!')
        return final_dir
    
    elif (sub_dir == '#'):
        del dir_tree[-1]
        final_dir = select_dir(os.sep.join(dir_tree))
    else:
        dir_tree.append(sub_dir)
        if not os.path.isdir(os.sep.join(dir_tree)):
            print('Invalid selection!')
            del dir_tree[-1]
        final_dir = select_dir(os.sep.join(dir_tree)) 
    return final_dir

"""
List out the details of the directory
"""
def list_dir(name):
    if not os.path.isdir(name):
        print(name, ' is not a valid directory!')
        return
    dirs = sorted(os.listdir(name))
    print('List of sub-directories and files: ')
    for tempdir in dirs: 
        temp = name + os.sep + tempdir
        if(os.path.isdir(temp)): 
            file_count, file_size = get_dir_details(temp)
            print(termcolor.colored('%s \t %s \t %i'%(tempdir, file_size, file_count), 'cyan'))
    for tempdir in dirs:
        temp = name + os.sep + tempdir
        if(os.path.isfile(temp)): 
            file_count, file_size = get_file_details(temp)
            print(termcolor.colored('%s \t %s'%(tempdir, file_size), 'magenta'))

"""
Get size and number of files for the given directory
"""
def get_dir_details(name):
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
def get_file_details(name):
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


#
generate_config()