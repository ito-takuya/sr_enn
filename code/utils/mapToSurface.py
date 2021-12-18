# Taku Ito
# 03/26/2019
import numpy as np
import nibabel as nib
import os

glasserfile2 = '/projects/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser2 = nib.load(glasserfile2).get_data()
glasser2 = np.squeeze(glasser2)

def mapToSurface(array,filename):
    """
    array can either be 360 array or ~59k array. If 360, will automatically map back to ~59k
    """
    #### Map back to surface
    if array.shape[0]==360:
        out_array = np.zeros((glasser2.shape[0],3))

        roicount = 0
        for roi in rois:
            for col in range(array.shape[1]):
                vertex_ind = np.where(glasser2==roi+1)[0]
                out_array[vertex_ind,0] = array[roicount,0]
                out_array[vertex_ind,1] = array[roicount,1]
                out_array[vertex_ind,2] = array[roicount,2]

            roicount += 1

    else:
        out_array = array

    #### 
    # Write file to csv and run wb_command
    np.savetxt(filename + '.csv', out_array,fmt='%s')
    wb_file = filename + '.dscalar.nii'
    wb_command = 'wb_command -cifti-convert -from-text ' + filename + '.csv ' + glasserfile2 + ' ' + wb_file + ' -reset-scalars'
    os.system(wb_command)
    os.remove(filename + '.csv')
