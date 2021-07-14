import numpy as np
import h5py
import tools
import nibabel as nib

# Excluding 084
subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033',
            '034','035','037','038','039','040','041','042','043','045','046','047','048','049','050',
            '053','055','056','057','058','062','063','066','067','068','069','070','072','074','075',
            '076','077','081','085','086','087','088','090','092','093','094','095','097','098','099',
            '101','102','103','104','105','106','108','109','110','111','112','114','115','117','119',
            '120','121','122','123','124','125','126','127','128','129','130','131','132','134','135',
            '136','137','138','139','140','141']

basedir = '/projects3/SRActFlow/'

# Using final partition
networkdef = np.loadtxt('/projects3/NetworkDiversity/data/network_partition.txt')
networkorder = np.asarray(sorted(range(len(networkdef)), key=lambda k: networkdef[k]))
networkorder.shape = (len(networkorder),1)
# network mappings for final partition set
networkmappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8, 'lan':6, 'dan':5, 'con':4, 'dmn':9, 'pmulti':10, 'none1':11, 'none2':12}
networks = networkmappings.keys()

## General parameters/variables
nParcels = 360
nSubjs = len(subjNums)

glasserfile2 = '/projects/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser2 = nib.load(glasserfile2).get_data()
glasser2 = np.squeeze(glasser2)


def main(roi_rh=9,roi_lh=189):
    for subj in subjNums:
        removeCollider(subj,roi_rh,roi_lh)

def removeCollider(subj,roi_rh,roi_lh):
    print 'Removing colliders for subject', subj, 'and ROIs', roi_rh, ',', roi_lh

    roi_lh_ind = np.where(glasser2==roi_lh)[0]
    roi_rh_ind = np.where(glasser2==roi_rh)[0]

    dilateLH = tools.loadMask(roi_lh,dilated=True)
    dilateRH = tools.loadMask(roi_rh,dilated=True)
    combinedDilated = dilateLH + dilateRH
    # combinedDilated = dilateRH
    # Exclude all SMN regions
    smn_rois = np.where(networkdef==networkmappings['smn'])[0]
    for x in smn_rois:
        roi_ind = np.where(glasser2==x)[0]
        combinedDilated[roi_ind]=1
    source_ind = np.where(combinedDilated==0)[0]

    print '\tLoad and re-map PCA_FC back to space...'


    fcmapping_rh, eigenvector_rh = tools.loadFCandEig(subj,roi_rh)
    fcmapping_lh, eigenvector_lh = tools.loadFCandEig(subj,roi_lh)

    # Map eigen-connectivity back to activity space
    fcmapping_rh = np.dot(fcmapping_rh.T,eigenvector_rh[:,:]).T
    fcmapping_lh = np.dot(fcmapping_lh.T,eigenvector_lh[:,:]).T

    print '\tLoad rest data...'
    # Load resting-state data -- remove colliders
    rest_ts = tools.loadRestActivity(subj)
    source_ts = rest_ts[source_ind,:]

    print '\tCompute covariance matrices and remove colliders...'
    # Remove colliders, RH
    target_ts = rest_ts[roi_rh_ind,:]
    cov_mat = np.dot(source_ts,target_ts.T)

    pos_weights = np.multiply(cov_mat > 0, fcmapping_rh > 0)
    neg_weights = np.multiply(cov_mat < 0, fcmapping_rh < 0)
    caus_graph = pos_weights + neg_weights

    fc_mapping_rh = np.multiply(fcmapping_rh,caus_graph)

    # Remove colliders, LH
    target_ts = rest_ts[roi_lh_ind,:]
    cov_mat = np.dot(source_ts,target_ts.T)
        
    pos_weights = np.multiply(cov_mat > 0, fcmapping_lh > 0)
    neg_weights = np.multiply(cov_mat < 0, fcmapping_lh < 0)
    caus_graph = pos_weights + neg_weights

    fc_mapping_lh = np.multiply(fcmapping_lh,caus_graph)

    #### Saving out to new h5 file
    fcdir = '/projects3/SRActFlow/data/results/pcaFC/'
    # roi_rh first
    filename = fcdir + 'TargetParcel' + str(roi_rh) + '_pcaFC_nozscore_noColliders.h5'
    h5f = h5py.File(filename,'a')
    h5f.create_dataset(subj + '/sourceToTargetmapping',data=fc_mapping_rh)
    h5f.close()
    # roi_lh second
    filename = fcdir + 'TargetParcel' + str(roi_lh) + '_pcaFC_nozscore_noColliders.h5'
    h5f = h5py.File(filename,'a')
    h5f.create_dataset(subj + '/sourceToTargetmapping',data=fc_mapping_lh)
    h5f.close()
