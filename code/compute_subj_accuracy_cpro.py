import numpy as np

subjaccuracy = {'013':0.7474489796,'014':0.7678571429,'015':'NA','016':0.7193877551,'017':0.7091836735,'018':0.9413265306,'019':'NA','020':'NA','021':0.8494897959,'022':'NA','023':0.8647959184,'024':0.8903061224,'025':0.8443877551,'026':0.9591836735,'027':0.8903061224,'028':0.9489795918,'029':'NA','030':0.8265306122,'031':0.9158163265,'032':0.8443877551,'033':0.6581632653,'034':0.862244898,'035':0.7448979592,'036':'NA','037':0.7587209302,'038':0.8698979592,'039':0.9642857143,'040':0.9617346939,'041':0.7888446215,'042':0.8903061224,'043':0.9056122449,'044':'NA','045':0.8903061224,'046':0.875,'047':0.7857142857,'048':0.8852040816,'049':0.8647959184,'050':0.8647959184,'051':'NA','052':'NA','053':0.8801020408,'054':'NA','055':0.9107142857,'056':0.8841201717,'057':0.8545918367,'058':0.8010204082,'059':'NA','060':0.8673469388,'061':'NA','062':0.9209183673,'063':0.8852040816,'064':0.770408163,'065':'NA','066':0.7423469388,'067':0.8392857143,'068':0.6607142857,'069':0.9183673469,'070':0.8086734694,'071':'NA','072':0.8010204082,'073':'NA','074':0.887755102,'075':0.8265306122,'076':0.8265306122,'077':0.8290816327,'078':0.806122449,'079':'NA','080':'NA','081':0.6836734694,'082':0.737244898,'083':'NA','084':'NA','085':0.8852040816,'086':0.8647959184,'087':0.9209183673,'088':0.9081632653,'089':'NA','090':0.829081633,'091':'NA','092':0.9413265306,'093':0.9260204082,'094':0.8469387755,'095':0.8698979592,'096':'NA','097':0.8392857143,'098':0.859693878,'099':0.9056122449,'100':'NA','101':0.9540816327,'102':0.8137755102,'103':0.9081632653,'104':0.8367346939,'105':0.7831632653,'106':0.8698979592,'107':'NA','108':0.8520408163,'109':0.7678571429,'110':0.887755102,'111':0.875,'112':0.8392857143,'113':'NA','114':0.7882653061,'115':0.8316326531,'116':'NA','117':0.7984693878,'118':0.5255102041,'119':0.8571428571,'120':0.6071428571,'121':0.7729591837,'122':0.9132653061,'123':0.9413265306,'124':0.7908163265,'125':0.8647959184,'126':0.887755102,'127':0.8469387755,'128':0.7602040816,'129':0.9183673469,'130':0.9056122449,'131':0.5102040816,'132':'NA','133':'NA','134':0.8545918367,'135':'NA','136':0.9515306122,'137':0.6479591837,'138':0.875,'139':0.6479591837,'140':0.6862244898,'141':0.8392857143}

subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033','034','035','037','038','039','040','041','042','043','045','046','047','048','049','050','053','055','056','057','058','062','063','066','067','068','069','070','072','074','075','076','077','081','085','086','087','088','090','092','093','094','095','097','098','099','101','102','103','104','105','106','108','109','110','111','112','114','115','117','119','120','121','122','123','124','125','126','127','128','129','130','131','132','134','135','136','137','138','139','140','141']

included_subj_acc = []

for subj in subjNums:
    if subjaccuracy[subj]=='NA':
        continue
    included_subj_acc.append(subjaccuracy[subj])

print(len(included_subj_acc))
print('Mean accuracy:', np.mean(included_subj_acc))
print('Median accuracy:', np.median(included_subj_acc))
print('Standard deviation of accuracy:', np.std(included_subj_acc))
print('Minimum accuracy:', np.min(included_subj_acc))
print('Maximum accuracy:', np.max(included_subj_acc))
