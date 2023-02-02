import os,sys
from doris.doris_stack.functions.ESD_functions import freadbk
from doris.doris_stack.main_code.resdata import ResData
import numpy as np

def get_slv_arr_shape(doris_stack_dir, date, sensor = 's1', swath='1', burst='1'):
    
    if sensor =='s1':
        file_dir = os.path.join(date, 'swath_'+swath, 'burst_'+burst)
    else:
        file_dir = date
    
    slv_resFilename = os.path.join(doris_stack_dir, file_dir,'slave.res')
    slv_res = ResData(slv_resFilename)
    
    l0 = int(slv_res.processes['resample']['First_line (w.r.t. original_master)'])
    lN = int(slv_res.processes['resample']['Last_line (w.r.t. original_master)'])
    p0 = int(slv_res.processes['resample']['First_pixel (w.r.t. original_master)'])
    pN = int(slv_res.processes['resample']['Last_pixel (w.r.t. original_master)'])
    
    Naz_res = lN-l0+1
    Nrg_res = pN-p0+1
    
    return (Naz_res, Nrg_res)


def get_cropped_image(choice_amp_int, doris_stack_dir, date, amp_file_name='slave_rsmp.raw', ifgs_file_name = 'cint_srd.raw' , dem_file_name = 'dem_radar.raw', atm_file_name = 'ref_atm.raw', crop_switch=False, crop_list=[], sensor='s1', swath='1', burst='1'):
    '''
    Function to read a subset of of slave_rsmp.raw or cint_srd.raw
    Args:
    
    
    '''
    if sensor =='s1':
        file_dir = os.path.join(date, 'swath_'+swath, 'burst_'+burst)
    else:
        file_dir = date
        
    file_name_dict = {'amp': amp_file_name,\
        'ifgs' : ifgs_file_name,\
            'dem' : dem_file_name,\
                'atm' : atm_file_name}
    file_dt_dir = {'amp': 'complex64',\
        'ifgs' : 'complex64',\
            'dem' : 'float32',\
                'atm' : 'float32'}
    
    file_name = file_name_dict[choice_amp_int]
    file_dt = file_dt_dir[choice_amp_int]
    
    dataFilename = os.path.join(doris_stack_dir, file_dir, file_name)
    #ifgs_dataFilename = os.path.join(doris_stack_dir, file_dir, ifgs_file_name)
    
    slv_resFilename = os.path.join(doris_stack_dir, file_dir,'slave.res')
    ifg_resFilename = os.path.join(doris_stack_dir, file_dir,'ifgs.res')
    
    slv_res = ResData(slv_resFilename)
    #ifg_res = ResData(ifg_resFilename)
    
    l0 = int(slv_res.processes['resample']['First_line (w.r.t. original_master)'])
    lN = int(slv_res.processes['resample']['Last_line (w.r.t. original_master)'])
    p0 = int(slv_res.processes['resample']['First_pixel (w.r.t. original_master)'])
    pN = int(slv_res.processes['resample']['Last_pixel (w.r.t. original_master)'])
    
    # Image size
    Naz_res = lN-l0+1
    Nrg_res = pN-p0+1
    if crop_switch:
        lines = crop_list[1]-crop_list[0]
        pixels = crop_list[3]-crop_list[2]
    else:
        lines = Naz_res
        pixels = Nrg_res
        crop_list = [1, Naz_res, 1, Nrg_res]
    
    print('Reading {} data from date {}. l0, p0, lines, pixels = {}, {}, {}, {}'.format(choice_amp_int, date, crop_list[0], crop_list[2], lines, pixels))
    
    arr = freadbk(dataFilename, 
                    crop_list[0]+1, 
                    crop_list[2], 
                    lines,pixels,
                    file_dt, int(Naz_res), int(Nrg_res))
    
    return arr
        
def get_dates(doris_stack_dir, master_date):
    #master_date = master_date.strftime("%Y%m%d")
    dates = sorted([l for l in [j for k,j,i in os.walk(doris_stack_dir)][0] if (len(l)==8)])
    dates.remove(str(master_date))
    return dates
    return [datetime.datetime.strptime(l, '%Y%m%d') for l in dates]

def get_stack(choice_amp_int, dates, doris_stack_dir, crop_switch=True, crop_list=[100,200,100,200], sensor='s1', swath='1', burst='1'):
    if crop_switch:
        lines = crop_list[1]-crop_list[0]
        pixels = crop_list[3]-crop_list[2]
    else:
        lines,pixels = get_slv_arr_shape(doris_stack_dir, dates[0])
    
    file_dt_dir = {'amp': 'complex64',\
        'ifgs' : 'complex64',\
            'dem' : 'float32',\
                'atm' : 'float32'}
    
    res = np.zeros((lines,pixels, len(dates)), dtype = file_dt_dir[choice_amp_int])#np.complex64)
    for i,date in enumerate(dates):
        slc_arr = get_cropped_image(choice_amp_int, doris_stack_dir, date, crop_switch=crop_switch, crop_list=crop_list, sensor='s1', swath='1', burst='1')
        #slc_arr = 10*np.log10(np.absolute(slc_arr))
        #plt.imshow(np.clip(slc_arr, 0, 35), cmap='gray')
        #plt.show()
        res[...,i] = slc_arr
    return res
