import os
import xlsxwriter
import pydicom

from natsort import natsorted

def readfileslist(pathtofiles, fileext):
    # Get filename from folder
    if not os.path.exists(pathtofiles):
        raise IOError('Path does not exist!')
        
    if not os.path.isabs(pathtofiles):
        pathtofiles = os.path.join(os.getcwd(), pathtofiles)

    lstFiles = []
    for dirName, _, filelist in os.walk(pathtofiles):
        for filename in natsorted(filelist):
            if fileext in filename.lower():
                lstFiles.append(os.path.join(dirName, filename))
    return lstFiles        

def writexlsx(df, filepath):
    # Write pandas dataframe to file with xlsxwriter lib
    colname = ['File_path', 'Offset_col (mm)', 'Offset_row (mm)', 'Tilt_angle (degree)', 'Is_offset', 'Pic_path'] # Need to adjust for different dataframe
    df_colname = ['File_path', 'Offset_col', 'Offset_row', 'Tilt_angle', 'Is_offset', 'Pic_path'] # Need to adjust for different dataframe
    
    workbook = xlsxwriter.Workbook(filepath)
    worksheet = workbook.add_worksheet()
    filepathlength = [len(filename) for filename in df['File_path']]
    worksheet.set_column(0, 0, max(filepathlength))
    for i in range(1, 4):
        worksheet.set_column(i, i, len(colname[i]))
    worksheet.set_column(len(df.columns) - 1, len(df.columns) - 1, max(filepathlength))
    
    redfontcolor = workbook.add_format({'font_color': 'red', 'num_format': '0.00', 'align': 'center'})        
    underlineformat = workbook.add_format({'underline': 1, 'font_color': 'blue', 'align': 'center'})
    normalformat = workbook.add_format({'num_format': '0.00', 'align': 'center'})
    
    for col, col_name in enumerate(colname):
        worksheet.write(0, col, col_name, normalformat)
       
    for col, col_name in enumerate(df_colname[0:-1]):
        for row, val in enumerate(df[col_name]):
            if df['Is_offset'][row] == 'Yes':
                worksheet.write(row + 1, col, val, redfontcolor)
            else:
                worksheet.write(row + 1, col, val, normalformat)
    
    for row, val in enumerate(df['Pic_path']):
        worksheet.write_url(row + 1, len(df.columns) - 1, 'external:' + val, underlineformat)
    
    workbook.close()
    
    return None             
    
def read_dicom_header(dcmpath):
    # Now only return the imager pixel spacing
    dcm = pydicom.read_file(dcmpath)
    
    return dcm.ImagerPixelSpacing
