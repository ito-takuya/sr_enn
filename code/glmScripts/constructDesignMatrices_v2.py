# Taku Ito
# 12/6/2018
# 
# Code to generate SR (context X stimulus integration regressors)
# Need to generate SR for each sensory rule

import numpy as np
import glob

#basedir = '/projects3/SRActFlow/'
basedir = '/home/ti61/f_mc1689_1/SRActFlow/'
stimdir = basedir + 'data/stimfiles/'


def srRED(subj):
    color_stims = stimdir + 'cproColorStim/'

    redred = glob.glob(color_stims + subj + '*REDRED.1D')
    redred = np.loadtxt(redred[0])
    
    redblue = glob.glob(color_stims + subj + '*REDBLUE.1D')
    redblue = np.loadtxt(redblue[0])

    bluered = glob.glob(color_stims + subj + '*BLUERED.1D')
    bluered = np.loadtxt(bluered[0])

    blueblue = glob.glob(color_stims + subj + '*BLUEBLUE.1D')
    blueblue = np.loadtxt(blueblue[0])

    # Load logic rules
    rules = loadLogic(subj)
    sensory = loadSensory(subj)
    red_rule = sensory[:,0]
    both = np.multiply(rules[:,0],red_rule)
    notboth = np.multiply(rules[:,1],red_rule)
    either = np.multiply(rules[:,2],red_rule)
    neither = np.multiply(rules[:,3],red_rule)

    # Concatenate, to generate appropriate regressors
    sr_red = np.vstack((both, notboth, either, neither)).T

    return sr_red

def srVERT(subj):
    ori_stims = stimdir + 'cproOriStim/'

    vertvert = glob.glob(ori_stims + subj + '*VERTICALVERTICAL.1D')
    vertvert = np.loadtxt(vertvert[0])
    
    verthori = glob.glob(ori_stims + subj + '*VERTICALHORIZONTAL.1D')
    verthori = np.loadtxt(verthori[0])

    horivert = glob.glob(ori_stims + subj + '*HORIZONTALVERTICAL.1D')
    horivert = np.loadtxt(horivert[0])

    horihori = glob.glob(ori_stims + subj + '*HORIZONTALHORIZONTAL.1D')
    horihori = np.loadtxt(horihori[0])

    # Load logic rules
    rules = loadLogic(subj)
    sensory = loadSensory(subj)
    vert_rule = sensory[:,1]
    both = np.multiply(rules[:,0],vert_rule)
    notboth = np.multiply(rules[:,1],vert_rule)
    either = np.multiply(rules[:,2],vert_rule)
    neither = np.multiply(rules[:,3],vert_rule)

    # Concatenate, to generate appropriate regressors
    sr_vert = np.vstack((both, notboth, either, neither)).T

    return sr_vert

def srHIGH(subj):
    pitch_stims = stimdir + 'cproPitchStim/'

    highhigh = glob.glob(pitch_stims + subj + '*HIGHHIGH.1D')
    highhigh = np.loadtxt(highhigh[0])
    
    highlow = glob.glob(pitch_stims + subj + '*HIGHLOW.1D')
    highlow = np.loadtxt(highlow[0])

    lowhigh = glob.glob(pitch_stims + subj + '*LOWHIGH.1D')
    lowhigh = np.loadtxt(lowhigh[0])

    lowlow = glob.glob(pitch_stims + subj + '*LOWLOW.1D')
    lowlow = np.loadtxt(lowlow[0])

    # Load logic rules
    rules = loadLogic(subj)
    sensory = loadSensory(subj)
    high_rule = sensory[:,2]
    both = np.multiply(rules[:,0],high_rule)
    notboth = np.multiply(rules[:,1],high_rule)
    either = np.multiply(rules[:,2],high_rule)
    neither = np.multiply(rules[:,3],high_rule)

    # Concatenate, to generate appropriate regressors
    sr_high = np.vstack((both, notboth, either, neither)).T

    return sr_high

def srCONSTANT(subj):
    constant_stims = stimdir + 'cproConstantStim/'

    constconst = glob.glob(constant_stims + subj + '*CONSTANTCONSTANT.1D')
    constconst = np.loadtxt(constconst[0])
    
    constalarm = glob.glob(constant_stims + subj + '*CONSTANTALARM.1D')
    constalarm = np.loadtxt(constalarm[0])

    alarmconst = glob.glob(constant_stims + subj + '*ALARMCONSTANT.1D')
    alarmconst = np.loadtxt(alarmconst[0])

    alarmalarm = glob.glob(constant_stims + subj + '*ALARMALARM.1D')
    alarmalarm = np.loadtxt(alarmalarm[0])

    # Load logic rules
    rules = loadLogic(subj)
    sensory = loadSensory(subj)
    constant_rule = sensory[:,3]
    both = np.multiply(rules[:,0],constant_rule)
    notboth = np.multiply(rules[:,1],constant_rule)
    either = np.multiply(rules[:,2],constant_rule)
    neither = np.multiply(rules[:,3],constant_rule)

    # Concatenate, to generate appropriate regressors
    sr_constant = np.vstack((both, notboth, either, neither)).T

    return sr_constant
    
def loadLogic(subj):
    logic_stims = stimdir + 'cproLogicRules/'

    both = glob.glob(logic_stims + subj + '*_BOTH.1D')
    both = np.loadtxt(both[0])

    notboth = glob.glob(logic_stims + subj + '*_NOTBOTH.1D')
    notboth = np.loadtxt(notboth[0])

    either = glob.glob(logic_stims + subj + '*_EITHER.1D')
    either = np.loadtxt(either[0])

    neither = glob.glob(logic_stims + subj + '*_NEITHER.1D')
    neither = np.loadtxt(neither[0])

    logicStimMat = np.vstack((both,notboth,either,neither)).T

    return logicStimMat

def loadSensory(subj):
    sensory_stims = stimdir + 'cproSensoryRules/'

    red = glob.glob(sensory_stims + subj + '*_RED.1D')
    red = np.loadtxt(red[0])

    vertical = glob.glob(sensory_stims + subj + '*_VERTICAL.1D')
    vertical = np.loadtxt(vertical[0])

    high = glob.glob(sensory_stims + subj + '*_HIGH.1D')
    high = np.loadtxt(high[0])

    constant = glob.glob(sensory_stims + subj + '*_CONSTANT.1D')
    constant = np.loadtxt(constant[0])

    sensoryStimMat = np.vstack((red,vertical,high,constant)).T

    return sensoryStimMat

def loadMotor(subj):
    motor_stims = stimdir + 'cproMotorRules/'

    lmid = glob.glob(motor_stims + subj + '*_LMID.1D')
    lmid = np.loadtxt(lmid[0])

    lind = glob.glob(motor_stims + subj + '*_LIND.1D')
    lind = np.loadtxt(lind[0])

    rmid = glob.glob(motor_stims + subj + '*_RMID.1D')
    rmid = np.loadtxt(rmid[0])

    rind = glob.glob(motor_stims + subj + '*_RIND.1D')
    rind = np.loadtxt(rind[0])

    motorStimMat = np.vstack((lmid,lind,rmid,rind)).T

    return motorStimMat 


def loadColorStim(subj):
    color_stims = stimdir + 'cproColorStim/'

    redred = glob.glob(color_stims + subj + '*REDRED.1D')
    redred = np.loadtxt(redred[0])
    
    redblue = glob.glob(color_stims + subj + '*REDBLUE.1D')
    redblue = np.loadtxt(redblue[0])

    bluered = glob.glob(color_stims + subj + '*BLUERED.1D')
    bluered = np.loadtxt(bluered[0])

    blueblue = glob.glob(color_stims + subj + '*BLUEBLUE.1D')
    blueblue = np.loadtxt(blueblue[0])

    colorStimMat = np.vstack((redred,redblue,bluered,blueblue)).T

    return colorStimMat

def loadOriStim(subj):
    ori_stims = stimdir + 'cproOriStim/'

    vertvert = glob.glob(ori_stims + subj + '*VERTICALVERTICAL.1D')
    vertvert = np.loadtxt(vertvert[0])
    
    verthori = glob.glob(ori_stims + subj + '*VERTICALHORIZONTAL.1D')
    verthori = np.loadtxt(verthori[0])

    horivert = glob.glob(ori_stims + subj + '*HORIZONTALVERTICAL.1D')
    horivert = np.loadtxt(horivert[0])

    horihori = glob.glob(ori_stims + subj + '*HORIZONTALHORIZONTAL.1D')
    horihori = np.loadtxt(horihori[0])

    oriStimMat = np.vstack((vertvert, verthori, horivert, horihori)).T

    return oriStimMat

def loadPitchStim(subj):
    pitch_stims = stimdir + 'cproPitchStim/'

    highhigh = glob.glob(pitch_stims + subj + '*HIGHHIGH.1D')
    highhigh = np.loadtxt(highhigh[0])
    
    highlow = glob.glob(pitch_stims + subj + '*HIGHLOW.1D')
    highlow = np.loadtxt(highlow[0])

    lowhigh = glob.glob(pitch_stims + subj + '*LOWHIGH.1D')
    lowhigh = np.loadtxt(lowhigh[0])

    lowlow = glob.glob(pitch_stims + subj + '*LOWLOW.1D')
    lowlow = np.loadtxt(lowlow[0])

    pitchStimMat = np.vstack((highhigh, highlow, lowhigh, lowlow)).T

    return pitchStimMat

def loadConstantStim(subj):
    constant_stims = stimdir + 'cproConstantStim/'

    constconst = glob.glob(constant_stims + subj + '*CONSTANTCONSTANT.1D')
    constconst = np.loadtxt(constconst[0])
    
    constalarm = glob.glob(constant_stims + subj + '*CONSTANTALARM.1D')
    constalarm = np.loadtxt(constalarm[0])

    alarmconst = glob.glob(constant_stims + subj + '*ALARMCONSTANT.1D')
    alarmconst = np.loadtxt(alarmconst[0])

    alarmalarm = glob.glob(constant_stims + subj + '*ALARMALARM.1D')
    alarmalarm = np.loadtxt(alarmalarm[0])

    constantStimMat = np.vstack((constconst, constalarm, alarmconst, alarmalarm)).T
    
    return constantStimMat

def loadMotorResponse(subj):
    motor_resp = stimdir + 'cproMotorResponse/'

    lmid = glob.glob(motor_resp + subj + '*_LMID.1D')
    lmid = np.loadtxt(lmid[0])
    
    lind = glob.glob(motor_resp + subj + '*_LINDEX.1D')
    lind = np.loadtxt(lind[0])

    rmid = glob.glob(motor_resp + subj + '*_RMID.1D')
    rmid = np.loadtxt(rmid[0])

    rind = glob.glob(motor_resp + subj + '*_RINDEX.1D')
    rind = np.loadtxt(rind[0])

    motorRespMat = np.vstack((lmid,lind,rmid,rind)).T

    return motorRespMat


def load64TaskEncoding(subj):
    context_stims = stimdir + 'cpro64TaskContext_Encoding/'

    timings = []
    files = glob.glob(context_stims + subj + '*_EV*.1D')
    ncontexts = len(files)
    for i in range(1,ncontexts+1):
        stimfile = glob.glob(context_stims + subj + '*_EV' + str(i) + '_Task' + str(i) + '.1D')
        timings.append(np.loadtxt(stimfile[0]))

    timings = np.asarray(timings).T

    return timings
