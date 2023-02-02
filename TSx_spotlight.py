#Extract and process tSx spotlight data
import os, tarfile, sys
from datetime import datetime
import numpy as np
from doris.doris_stack.main_code.jobs import Jobs
from doris.doris_stack.main_code.resdata import ResData
import warnings
from collections import Counter, OrderedDict
from para_jobs import Para_Jobs

def export_files(tar_dir):
    tar_files = [k for i,j,k in os.walk(tar_dir)][0]
    for tar_file in tar_files:
        tar = tarfile.open(os.path.join(dir, tar_file), "r:gz")
        members = tar.getmembers()
        
        #files = tar.getnames()
        #xml_file = [i for i in files if (((i.split('/')[-1][:3]=='TDX') | (i.split('/')[-1][:3]=='TSX')) & (i.split('/')[-1][-4:]=='.xml'))]
        
        #for count, file in enumerate(files):
        for member in tar.getmembers():
            file = member.name
            if len(file.split('/'))==4:
                tsx_filename = file.split('/')[-2]
                if (((tsx_filename[:3]=='TDX') | (tsx_filename[:3]=='TSX')) & (file.split('/')[-1][-4:]!='.xml')):
                    #print(file)
                    #continue
                    date = tsx_filename[28:36]
                    print(date)
                    date_obj = datetime.strptime(date, '%Y%m%d')
                    file_path = '/'.join(file.split('/')[:-1])
                    if ((date_obj >= datetime.strptime(start_date, '%Y%m%d')) & (date_obj <= datetime.strptime(end_date, '%Y%m%d'))):
                        print(file_path + ' selected. Exporting file ')
                        tar.extractall(path = export_path)
                    else:
                        print(file_path + ' : date out of range')

def make_folder_stack(export_path, doris_stack_dir):
    folder_list = [j for i,j,k in os.walk(export_path)][0]
    date_list = [folder[28:36] for folder in folder_list]
    stack_folder_list = [os.path.join(doris_stack_dir, date) for date in date_list]
    [os.mkdir(stack_folder) for stack_folder in stack_folder_list if not os.path.exists(stack_folder)]
                
                
def dump_slave_data(path, doris_stack_dir, folder_list, date_list, stack_folder_list, xml_file_list):
    
    for folder, xml_file, stack_folder in zip(folder_list, xml_file_list, stack_folder_list):
        xml_file_path = os.path.join(path, folder, xml_file)
        
        #dump header to doris
        os.system('python /home/anurag/Documents/PhDProject/doris/bin/tsx_dump_header2doris.py {} > {}/slave.res'.format(xml_file_path, stack_folder))
        
        cos_file_name = [k for i,j,k in os.walk(os.path.join(path, folder, 'IMAGEDATA'))][0][0]
        #Dump data
        os.system('python2 /home/anurag/Documents/PhDProject/doris/bin/tsx_dump_data.py {} {} -res {}/slave.res'.format(os.path.join(path, folder, 'IMAGEDATA',cos_file_name), os.path.join(stack_folder, 'slave.raw'), stack_folder))
       

def link_master(doris_stack_dir, date_list, master_date):
    #date_list.remove(master_date)
    #stack_folder_list
    #copy master res file
    master_res_orig = os.path.join(doris_stack_dir, master_date, 'slave.res')
    master_raw_orig = os.path.join(doris_stack_dir, master_date, 'slave.raw')
    
    for date in date_list:
        master_res = os.path.join(doris_stack_dir, date, 'master.res')
        master_raw = os.path.join(doris_stack_dir, date, 'master.raw')
        
        if not os.path.exists(master_res):
            os.system('cp {} {}'.format(master_res_orig, master_res))
            res = ResData(master_res)
            res.processes['crop']['Data_output_file'] = 'master.raw'
            res.write(new_filename = master_res)
        
        if not os.path.exists(master_raw):
            os.symlink(master_raw_orig, master_raw)
                  
                  
def coarse_orbits(date_list, doris_stack_dir, master_date, input_file_dir):
    
    joblist=[]
    for date in date_list:
        os.chdir(os.path.join(doris_stack_dir, date))
        os.system('doris {}/input.coarseorb'.format(input_file_dir))

    
def fake_fine_corr(date_list, master_date, stack_folder_list):
    
    coreg = OrderedDict()
    coreg['Initial offsets (l,p)'] = '0, 0'
    coreg['Window_size_L_for_correlation'] = '64'
    coreg['Window_size_P_for_correlation'] = '64'
    coreg['Max. offset that can be estimated'] = '32'
    coreg['Peak search ovs window (l,p)'] = '16 , 16'
    coreg['Oversampling factor'] = '32'
    coreg['Number_of_correlation_windows'] = '0'
    
    
    for date in date_list:
        
        ifg_res = os.path.join(doris_stack_dir, date, 'ifgs.res')
        res = ResData(ifg_res)
        print(date)
        if not res.process_control['fine_coreg'] == '1':
            res.insert(coreg,'fine_coreg')
            res.write(new_filename = ifg_res)

def dem_assist(date_list, master_date, input_files_path, para_jobs_obj):
    jobList1 = []
    
    for date in date_list:
        ifg_res = os.path.join(doris_stack_dir, date, 'ifgs.res')
        res = ResData(ifg_res)
        if not res.process_control['dem_assist'] == '1':
            path = os.path.join(doris_stack_dir, date)
            command1 = 'doris '+ os.path.join(input_files_path, 'input.dembased')
            jobList1.append({"path": path, "command": command1})

    
        jobs = Jobs(15, para_jobs_obj)
        jobs.run(jobList1)


def fake_coregpm(doris_stack_dir):
    
    coreg = OrderedDict()
    coreg['Degree_cpm'] = '0'
    coreg['Normalization_Lines'] = ''
    coreg['Normalization_Pixels'] = ''
    coreg['Estimated_coefficientsL'] = ''
    coreg['row_0'] = ["{0:.8e}".format(0), '0', '0']
    coreg['Estimated_coefficientsP'] = ''
    coreg['row_1'] = ["{0:.8e}".format(0), '0', '0']

    coreg['Deltaline_slave00_poly'] = "{0:.8e}".format(0)
    coreg['Deltapixel_slave00_poly'] = "{0:.8e}".format(0)
    coreg['Deltaline_slave0N_poly'] = "{0:.8e}".format(0)
    coreg['Deltapixel_slave0N_poly'] = "{0:.8e}".format(0)
    coreg['Deltaline_slaveN0_poly'] = "{0:.8e}".format(0)
    coreg['Deltapixel_slaveN0_poly'] = "{0:.8e}".format(0)
    coreg['Deltaline_slaveNN_poly'] = "{0:.8e}".format(0)
    coreg['Deltapixel_slaveNN_poly'] = "{0:.8e}".format(0)
    
    for date in date_list:
        ifg_res_file = os.path.join(doris_stack_dir, date, 'ifgs.res')
        mas_res_file = os.path.join(doris_stack_dir, date, 'master.res')
        
        ifg_res = ResData(ifg_res_file)
        mas_res = ResData(mas_res_file)
        
        lines = (int(mas_res.processes['crop']['Last_line (w.r.t. original_image)']) -
                        int(mas_res.processes['crop']['First_line (w.r.t. original_image)']))
        pixels = (int(mas_res.processes['crop']['Last_pixel (w.r.t. original_image)']) -
                    int(mas_res.processes['crop']['First_pixel (w.r.t. original_image)']))

        # Save pixels lines
        coreg['Normalization_Lines'] = "{0:.8e}".format(1) + ' ' + "{0:.8e}".format(lines)
        coreg['Normalization_Pixels'] = "{0:.8e}".format(1) + ' ' + "{0:.8e}".format(pixels)

        # Copy coregistration from full swath to burst
        if not ifg_res.process_control['comp_coregpm'] == '1':
            ifg_res.insert(coreg,'comp_coregpm')
            ifg_res.write(new_filename = ifg_res_file)


def resample(date_list, master_date, input_file_path, para_jobs_obj):
    jobList1 = []
    
    for date in date_list:
        slv_res = os.path.join(doris_stack_dir, date, 'slave.res')
        res = ResData(slv_res)
        if not res.process_control['resample'] == '1':
            path = os.path.join(doris_stack_dir, date)
            command1 = 'doris '+ os.path.join(input_file_path, 'input.resample')
            jobList1.append({"path": path, "command": command1})

    
        jobs = Jobs(15, para_jobs_obj)
        jobs.run(jobList1)   
        
def interferogram(date_list, master_date, input_file_path, para_jobs_obj):
    jobList1 = []
    
    for date in date_list:
        ifg_res = os.path.join(doris_stack_dir, date, 'ifgs.res')
        res = ResData(ifg_res)
        if not res.process_control['interfero'] == '1':
            path = os.path.join(doris_stack_dir, date)
            command1 = 'doris '+ os.path.join(input_file_path, 'input.interferogram')
            jobList1.append({"path": path, "command": command1})

    
        jobs = Jobs(15, para_jobs_obj)
        jobs.run(jobList1) 


   
def comp_refphase(date_list, master_date, input_file_path, para_jobs_obj):
    jobList1 = []
    
    for date in date_list:
        ifg_res = os.path.join(doris_stack_dir, date, 'ifgs.res')
        res = ResData(ifg_res)
        if not res.process_control['comp_refphase'] == '1':
            path = os.path.join(doris_stack_dir, date)
            command1 = 'doris '+ os.path.join(input_file_path, 'input.comprefpha')
            jobList1.append({"path": path, "command": command1})

    
        jobs = Jobs(15, para_jobs_obj)
        jobs.run(jobList1) 


def subt_ref_phase(date_list, master_date, input_file_path, para_jobs_obj):
    jobList1 = []
    
    for date in date_list:
        ifg_res = os.path.join(doris_stack_dir, date, 'ifgs.res')
        res = ResData(ifg_res)
        if not res.process_control['subtr_refphase'] == '1':
            path = os.path.join(doris_stack_dir, date)
            command1 = 'doris '+ os.path.join(input_file_path, 'input.subtrrefpha')
            jobList1.append({"path": path, "command": command1})

    
        jobs = Jobs(15, para_jobs_obj)
        jobs.run(jobList1)

def comp_refdem(date_list, master_date, input_file_path, para_jobs_obj):
    jobList1 = []
    
    for date in date_list:
        ifg_res = os.path.join(doris_stack_dir, date, 'ifgs.res')
        res = ResData(ifg_res)
        if not res.process_control['comp_refdem'] == '1':
            path = os.path.join(doris_stack_dir, date)
            command1 = 'doris '+ os.path.join(input_file_path, 'input.comprefdem')
            jobList1.append({"path": path, "command": command1})

        jobs = Jobs(15, para_jobs_obj)
        jobs.run(jobList1)

def return_to_process(date_list, master_date, method):
    
    for date in date_list:
        os.system('/home/anurag/Documents/PhDProject/doris/bin/doris.rmstep.sh '+ method +' '+ os.path.join(doris_stack_dir, date, 'ifgs.res'))

def make_sink_animation():
    pass
            
if __name__=='__main__':
    tar_dir = '/home/anurag/Documents/PhDProject/TS-X_wink_data/SLC_data'
    Jobs.id=0
    
    
    start_date = '20151001'
    end_date = '20160401'
    master_date  = '20151221'#'20151005'#'20151210'

    export_path = '/home/anurag/Documents/PhDProject/TS-X_wink_data/sinkhole_epochs'
    
    #doris_stack_dir = '/media/anurag/SSD_1/anurag/PhD_Project/Doris_Processing/Doris_Processing_35_wink_spotlight/stack'
    doris_stack_dir = '/media/anurag/Seagate_badi_vaali/PhDProject/Doris_Processing/Doris_Processing_35_wink_TSx/stack'
    
    input_file_path = os.path.join(os.path.dirname(doris_stack_dir), 'input_files')
    
    folder_list = [j for i,j,k in os.walk(export_path)][0]
    date_list = [folder[28:36] for folder in folder_list]
    stack_folder_list = [os.path.join(doris_stack_dir, date) for date in date_list]
    xml_file_list = [i+'.xml' for i in folder_list]
    
    para_jobs_obj = Para_Jobs(doris_stack_dir)
    
    #date_list.remove(master_date)
    
    print(np.array([int(i) for i in sorted(date_list)]))
    
    print('Making folders')
    make_folder_stack(export_path, doris_stack_dir)
    print('dumping slave data')
    dump_slave_data(export_path, doris_stack_dir, folder_list, date_list, stack_folder_list, xml_file_list)
    print('linking master')
    link_master(doris_stack_dir, date_list, master_date)
    print('Coarse orbits')
    coarse_orbits(date_list, doris_stack_dir, master_date, input_file_path)
    print('Faking fine graduation')
    fake_fine_corr(date_list, master_date, stack_folder_list)
    print('DAC')
    dem_assist(date_list, master_date, input_file_path, para_jobs_obj)
    print('Faking Coreg PM')
    fake_coregpm(doris_stack_dir)
    print('Resampling')
    resample(date_list, master_date, input_file_path, para_jobs_obj)
    print('interferogram')
    interferogram(date_list, master_date, input_file_path, para_jobs_obj)
    print('Comp ref phase')
    comp_refphase(date_list, master_date, input_file_path, para_jobs_obj)
    print('Subt. ref phase')
    subt_ref_phase(date_list, master_date, input_file_path, para_jobs_obj)
    print('Comp ref dem')
    comp_refdem(date_list, master_date, input_file_path, para_jobs_obj)
    return_to_process(date_list, master_date, 'subtr_refphase')
