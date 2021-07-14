# Taku Ito
# 12/6/2018
# 
# Code to generate SR (context X stimulus integration regressors)
# Need to generate SR for each sensory rule

import numpy as np
import glob

basedir = '/projects3/SRActFlow/data/'
stimdir = basedir + 'stimfiles/'


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

    ## Design regressors for correct logic computation of the RED rule 
    # EX: BOTH-RED: the intersection of BOTH blocks + REDRED stimulus pairing
    both_red = np.multiply(both,redred)
    notboth_red = np.multiply(notboth,blueblue) + np.multiply(notboth,bluered) + np.multiply(notboth,redblue)
    either_red = np.multiply(either,bluered) + np.multiply(either,redblue) + np.multiply(either,redred)
    neither_red = np.multiply(neither,blueblue)

    # Concatenate, to generate appropriate regressors
    sr_red = np.vstack((both_red, notboth_red, either_red, neither_red)).T

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

    ## Design regressors for correct logic computation of the RED rule 
    # EX: BOTH-RED: the intersection of BOTH blocks + vertvert stimulus pairing
    both_vert = np.multiply(both,vertvert)
    notboth_vert = np.multiply(notboth,horihori) + np.multiply(notboth,horivert) + np.multiply(notboth,verthori)
    either_vert = np.multiply(either,horivert) + np.multiply(either,verthori) + np.multiply(either,vertvert)
    neither_vert = np.multiply(neither,horihori)

    # Concatenate, to generate appropriate regressors
    sr_vert = np.vstack((both_vert, notboth_vert, either_vert, neither_vert)).T

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

    ## Design regressors for correct logic computation of the RED rule 
    # EX: BOTH-RED: the intersection of BOTH blocks + highhigh stimulus pairing
    both_high = np.multiply(both,highhigh)
    notboth_high = np.multiply(notboth,lowlow) + np.multiply(notboth,lowhigh) + np.multiply(notboth,highlow)
    either_high = np.multiply(either,lowhigh) + np.multiply(either,highlow) + np.multiply(either,highhigh)
    neither_high = np.multiply(neither,lowlow)

    # Concatenate, to generate appropriate regressors
    sr_high = np.vstack((both_high, notboth_high, either_high, neither_high)).T

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

    ## Design regressors for correct logic computation of the RED rule 
    # EX: BOTH-RED: the intersection of BOTH blocks + REDRED stimulus pairing
    both_constant = np.multiply(both,constconst)
    notboth_constant = np.multiply(notboth,alarmalarm) + np.multiply(notboth,alarmconst) + np.multiply(notboth,constalarm)
    either_constant = np.multiply(either,alarmconst) + np.multiply(either,constalarm) + np.multiply(either,constconst)
    neither_constant = np.multiply(neither,alarmalarm)

    # Concatenate, to generate appropriate regressors
    sr_constant = np.vstack((both_constant, notboth_constant, either_constant, neither_constant)).T

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
