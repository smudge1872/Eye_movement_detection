
import configparser
import os


from PatientDataset import PatientDataset

from util.util import is_correct_label, avg
from models.LSTMClassifier_alt import LSTMClassifier_alt
from models.OneDConv import OneDConv
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader


import matplotlib



matplotlib.use('Agg')


np.set_printoptions(suppress=True)



import pickle

import argparse

def writeCSVFile(patientNums, timeLabels, all_actual, all_pred, all_probs, outputFileName):

    all_probs = all_probs[:, 1]

    should_write_header = True

    print('Write out csv file %s' % (outputFileName))

    file = open(outputFileName, 'w+')
    # writer = csv.writer(file)
    if should_write_header == True:
        # writer.writerow(['patientNum','timeStamp(HH:MM:SS)','GT movement', 'Predict Movement', 'Movement probability'])
        file.write('patientNum,timeStamp(HH:MM:SS),Predict_Movement,Movement_probability\n')

    print('patientNum\ttimeStamp(HH:MM:SS)\tPredict_Movement\tMovement_probability')

    uniquePatNums = np.unique(patientNums)

    for patNum in uniquePatNums:

        #Sort by seconds
        maskPat = patientNums == patNum
        all_probsPat = all_probs[maskPat]
        all_predPat = all_pred[maskPat]
        all_actualPat = all_actual[maskPat]
        patientNumLabelsPat = patientNums[maskPat]
        timeLabelsPat = timeLabels[maskPat, :]
        secondLabelsPat = np.zeros(len(all_predPat))

        secondDict = {}

        for t in range(0, len(all_predPat)):
            secondLabelsPat[t] = timeLabelsPat[t, 2] + (60 * timeLabelsPat[t, 1]) + (60 * 60 * timeLabelsPat[t, 0])
            secondDict[secondLabelsPat[t]] = [patientNumLabelsPat[t], all_predPat[t], timeLabelsPat[t, :], all_probsPat[t]]

        sortedSecs = sorted(secondDict.keys())

        midnightIndex = -1

        for t in range(0, len(sortedSecs) - 1):
            if sortedSecs[t + 1] - sortedSecs[t] > 1500:
                print("Possible MidnightCrossing. The seconds are not sorted correctly for patient %d at second %d. Will try to correct" % (
                    patNum, t + 1))
                midnightIndex = t + 1
                break

        sortedSecNump = np.array(sortedSecs)

        if midnightIndex != -1:
            tmpEarlier = sortedSecNump[midnightIndex:]
            tmpLtr = sortedSecNump[0:midnightIndex]
            sortedSecNump = np.concatenate((tmpEarlier, tmpLtr))

        for secInd in range(0, len(sortedSecNump)):

            sec = sortedSecNump[secInd]

            info = secondDict[sec]
            #secondDict[secondLabelsPat[t]] = [patientNumLabelsPat[t], all_predPat[t], timeLabelsPat[t, :], all_probsPat[t]]
            file.write('%d,%02d:%02d:%02d,%d,%f\n' % (
            info[0], info[2][0], info[2][1], info[2][2],
            info[1], info[3]))
            # writer.writerow()
            print('%d\t%02d:%02d:%02d\t%d\t%f' % ( info[0], info[2][0], info[2][1], info[2][2],
            info[1], info[3]) )

    file.close()

    print(' ')


def validate(val_dataloader, model, device, loss_func, num_classes):
    model.eval()
    all_pred = []
    all_actual = []
    val_losses = []
    val_accs = []
    print_metrics_freq = np.floor(len(val_dataloader) / 10)
    all_probs = np.empty((0, num_classes))

    print('\nVALIDATION [', end='', flush=True)
    for i_batch, sample_batched in enumerate(val_dataloader):
        input_sequence = sample_batched[0].float()
        input_sequence = torch.squeeze(input_sequence)
        input_sequence = input_sequence.to(device)
        labels = sample_batched[1].to(device).squeeze().long()

        # When 1 wave feature used, shape only has 2 dimensions needs 3.
        if len(input_sequence.shape) != 3:
            input_sequence = input_sequence.view(input_sequence.shape[0], input_sequence.shape[1], 1)

        # if the last batch has a size of 1
        if len(labels.shape) == 0:
            input_sequence = input_sequence.view(1, input_sequence.shape[0], input_sequence.shape[1])
            labels = labels.view(1)

        y = model(input_sequence)
        val_losses.append(loss_func(y.float(), labels.long()).mean().item())

        # Compute classification accuracy for each output
        val_accs.extend(is_correct_label(y, labels))
        if i_batch % print_metrics_freq == 0:
            print('.', end='', flush=True)
        p = y.detach().cpu().numpy()
        ll = labels.detach().cpu().numpy()

        all_probs = np.concatenate((all_probs, p,), axis=0)

        all_pred.extend(np.argmax(p, axis=1))
        all_actual.extend(ll)
    else:
        print(']\n')
    all_pred = np.stack(all_pred)
    all_actual = np.stack(all_actual)
    return all_pred, all_actual, val_losses, val_accs, all_probs




def runModel(configFile):
    config = configparser.ConfigParser()

    config.read(configFile)

    inputFolderName = config['DEFAULT']['inputFolderName'];
    modelFileName = config['DEFAULT']['modelFileName'];

    rootOutputDir = '%s' % (config['DEFAULT']['outputdir'])

    norm_max = float(config['DEFAULT']['norm_max']);
    norm_min = float(config['DEFAULT']['norm_min']);
    norm_mn = float(config['DEFAULT']['norm_mn']);
    norm_stdv = float(config['DEFAULT']['norm_stdv']);
    lr = float(config['DEFAULT']['lr']);
    epochs = int(config['DEFAULT']['epochs'])
    num_features = int(config['DEFAULT']['num_features'])  # 9  # Defined by the data
    condense_method = int(config['DEFAULT']['condense_method'])  # 2 = 2 class(move/nomove)  3 = 3 class , 5 for 5 class
    # Binary case (movement, no movement)
    num_labels = int(config['DEFAULT']['num_labels'])  # 2 or 3

    batch_size = int(config['DEFAULT']['batch_size'])
    model_choice = int(config['DEFAULT']['model_choice'])
    class_type = int(config['DEFAULT']['class_type'])
    saveAllEpochModels = int(config['DEFAULT']['saveAllEpochModels'])
    sensToKeep = eval(config['DEFAULT']['sensToKeep'])
    seq_len = int(config['DEFAULT']['seq_len'])


    print('rootOutputDir is %s' % rootOutputDir)
    print('norm_max is %f' % norm_max)
    print('norm_min is %f' % norm_min)
    print('norm_mn is %f' % norm_mn)
    print('norm_stdv id %f' % norm_stdv)
    print('epochs is %d' % epochs)
    print('num_features is %d' % num_features)
    print('condense_method is %d' % condense_method)
    print('num_labels = %d' % num_labels)
    print('model choice is =%d' % model_choice)
    print('Training Batch size is %d' % batch_size)
    print('Class type = %d' % class_type)
    print('Save all epoch models = %d' % saveAllEpochModels)
    print('Sensors used %s, [] means all' % (config['DEFAULT']['sensToKeep']))
    print('seq_len = %d' % (seq_len))



    try:
        os.makedirs(rootOutputDir)
    except OSError as error:
        print(error)


    hold_in_memory = True  # Keep examples in memory. This is a small amount of memory, should be fine.

    clip = 3  # gradient clipping

    num_layers = 1  # Number of lstm's to stack on one another
    layer_size = 24  # Hidden cell size

    # Waveform feature normalization.
    # These values were gathered manually, more details in PatientDataset.
    use_feat_norm = True

    device = 'cuda'
    # device = 'cpu'

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)

    if model_choice == 1:
        print('Model is LSTM')
        model = LSTMClassifier_alt(num_features=num_features, batch_size=batch_size,
                                   seq_len=seq_len,
                                   num_layers=num_layers, layer_size=layer_size,
                                   num_labels=num_labels, device=device)
    elif model_choice == 2:
        print('Model is 1D convolution from sleep classification')
        model = OneDConv(num_features, seq_len, num_labels)



    model.float()
    model.to(device)



    test_dataset = PatientDataset(inputFolderName, sub_dirs=None, condense_outputs=condense_method,
                                      hold_in_memory=hold_in_memory,
                                      norm_feats=use_feat_norm,
                                      norm_max=norm_max,
                                      norm_min=norm_min,
                                      norm_mn=norm_mn,
                                      norm_stdv=norm_stdv, colsToKeep=sensToKeep)

    if class_type == 2:
        test_dataset.setDataForSpeedClassification()

    if class_type == 3:
        test_dataset.setDataForDirectionClassification()

    # Create loss functions and optimizer.
    weighted_loss_function = nn.CrossEntropyLoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Load model with best AUC on validation set

    bestThresh = 0.5

    bestModelPath = os.path.join(modelFileName)
    model_state_dict = torch.load(bestModelPath)['model_state_dict']
    model.load_state_dict(model_state_dict)

    model.float()
    model.to(device)

    # dataloader for testing on test
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    all_pred, all_actual, test_losses, test_accs, all_probs = validate(test_dataloader, model,
                                                                       device, weighted_loss_function, num_labels)

    timeLabels = test_dataset.getTimeLabel()
    patientNumLabels = test_dataset.getPatNumLabel()
    speedLabels = test_dataset.getSpeedLabel()
    dirLabels = test_dataset.getDirLabelMask()

    # Write out all the fold predictions in pickle file
    with open(os.path.join(rootOutputDir, 'TestPredOutput.pkl'), 'wb') as dfile:
        terms = pickle.dump([all_actual, all_probs, all_pred, patientNumLabels, timeLabels, speedLabels, dirLabels],
                            dfile)
        dfile.close()

    writeCSVFile(patientNumLabels, timeLabels, all_actual, all_pred, all_probs, os.path.join(rootOutputDir, 'TestOutput.csv'))






# MAIN
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=' train for eye movement')

    parser.add_argument('--configFile', required=False, default="./crack.data",
                        metavar="config ini file",
                        help='config ini file')


    args = parser.parse_args()

    configFile = args.configFile


    print('configFile is %s' % configFile)


    runModel(configFile=configFile)

    print('All done!')






