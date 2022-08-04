import h5py
import numpy as np
import glob
import os
import sys
import torch
from torch.utils.data.dataset import Dataset
from util import H5Reader
from util.util import condense_outputs3class
from util.util import condense_outputs5class
from util.util import condense_outputs_movement as condense_movement


class PatientDataset(Dataset):

    _H5_EXAMPLE_NAME = 'example'
    _H5_GT_NAME = 'gt'


    def __init__(self, data_dir, sub_dirs=None, transform=None, condense_outputs=0, hold_in_memory=True, norm_feats=True,
                    norm_max=3200, norm_min=-3200, norm_mn=0, norm_stdv=30, colsToKeep=[] ):
        # TODO: Get the minimal set of data normalization terms and put them in in a better place (within the code).
        # 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Notes on the normalization stats:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #  
        #  [Gathered Feb. 5 on patients 1-25 using <_cal_data_statistics()> with norm_feats=false]
        #  
        # 
        #  Currently there are 9 features in use. They seem to have a set range of
        #  [-3200, 3200], with most having a mn / stdv of 0 / 0~30.   
        #
        #  -3200.0 -3200.0 -3200.0 -3200.0 -3200.0 -3200.0 -3200.0 -3200.0 -3200.0
        #  3154.1   3163.0  3189.0  3134.7  3194.2  3171.8  3181.0  3174.9  3194.6
        #
        #  Normalization is a simple range reduction to [-1, 1] at the moment. 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self._transform = transform
        self._condense_outputs = condense_outputs
        
        self.norm_feats = norm_feats
        self.norm_max = norm_max
        self.norm_min = norm_min
        self.norm_mn = norm_mn
        self.norm_stdv = norm_stdv
        if len(colsToKeep) != 0:
            colsToKeep.sort()
        self.colsToKeep = colsToKeep

        if sub_dirs is None:
            self._data_files = glob.glob(os.path.join(data_dir, '*'))
        else:
            # Expect subdirs to be relative to data_dir root
            self._data_files = [x for sub_dir in sub_dirs for x in glob.glob(os.path.join(data_dir, sub_dir, '*'))]

        self._num_examples = len(self._data_files)

        # TODO: Programmatically determine this
        self._seq_len = 1
        
        self.hold_in_memory = hold_in_memory
        if self.hold_in_memory:
          self._initialized = False
          self._examples = {}
          self.binary_label_mask = []
          self.patNum_label_mask = []
          self.time_label_mask = []
          self.move_label_mask = []
          self.speed_label_mask = []
          self.dir_label_mask = []
          self._populate()

        # Inspect first example to determine number of outputs
        _, gt = self[0]
        self._label_dim = gt.shape[0]

        #if self._condense_outputs > 2:
        #    self._clean_data()
        
        #self._cal_data_statistics()
        

    def _populate(self):        
        print('Populating [', end='', flush=True)
        incro = np.floor(self._num_examples / 60)

        patNum_label_mask = np.zeros((self._num_examples), dtype=np.ubyte)
        time_label_mask = np.zeros((self._num_examples,3), dtype=np.ubyte)
        move_label_mask = np.zeros((self._num_examples), dtype=np.ubyte)
        speed_label_mask = np.zeros((self._num_examples), dtype=np.ubyte)
        dir_label_mask = np.zeros((self._num_examples), dtype=np.ubyte)


        if self._condense_outputs == 2:
            binary_label_mask = np.zeros(self._num_examples, dtype=bool)
        else:
            binary_label_mask = np.zeros(self._num_examples, dtype=np.ubyte)
        for i in range(0, self._num_examples):
          if i % incro == 0:
            print('.', end='', flush=True)            
          item = self.__getitem__(i)
          self._examples[i] = item
          gt = item[1]
          if self._condense_outputs == 2:
            binary_label_mask[i] = gt == 1
          else:
            binary_label_mask[i] = gt

          itemPatNum = self.getItemPatNum(i)

          patNum_label_mask[i] = itemPatNum

          itemTimeLab = self.getItemTime(i)

          gtVec = self.getItemGTVec(i)
          move_label_mask[i] = gtVec[0]
          speed_label_mask[i] = gtVec[1]
          dir_label_mask[i] = gtVec[2]

          for el in range(0, len(itemTimeLab)):
              time_label_mask[i, el] = itemTimeLab[el]


        print('] Done\n', flush=True)
        if self._condense_outputs == 2:
            print("Positive class count = %d" % np.sum(binary_label_mask))
            print("Negative class count = %d" % np.sum(np.logical_not(binary_label_mask)), flush=True)
        elif self._condense_outputs == 3 :
            print("0 class count = %d" % (np.sum(binary_label_mask == 0)  )  )
            print("1 class count = %d" % (np.sum(binary_label_mask == 1)  ) )
            print("2 class count = %d" % (np.sum(binary_label_mask == 2)), flush=True)
        elif self._condense_outputs == 5:
            for i in range(0,5):
                print("%d class count = %d" % (i, np.sum(binary_label_mask == i)), flush=True )


        self._initialized = True
        self.binary_label_mask = binary_label_mask
        self.patNum_label_mask = patNum_label_mask
        self.time_label_mask = time_label_mask
        self.move_label_mask = move_label_mask
        self.speed_label_mask = speed_label_mask
        self.dir_label_mask = dir_label_mask

    def setDataForDirectionClassification(self):
        #maskToKeep = np.ones(self._num_examples, dtype=bool)
        #maskToKeep[:] = False

        print('Filtering dataset for Direction classification only')

        maskToKeep = (self.move_label_mask == 1) & (self.dir_label_mask >= 1)

        filt_examples = {}
        new_numExamples = np.sum(maskToKeep)
        new_dataFiles = []


        sampsIndToInclude = np.nonzero(maskToKeep == True)

        newIndex = 0

        numPosExamps = 0
        numNegExamps = 0

        for kInd in sampsIndToInclude[0]:
            item = list(self._examples[kInd])
            if self.dir_label_mask[kInd] == 1:
                item[1] = np.zeros(1, dtype=np.float64)
                numNegExamps = numNegExamps + 1
            elif self.dir_label_mask[kInd] == 2:
                item[1] = np.ones(1, dtype=np.float64)
                numPosExamps = numPosExamps + 1

            self.binary_label_mask[kInd] = item[1] > 0
            filt_examples[newIndex] = tuple(item)
            new_dataFiles.append(self._data_files[kInd])
            newIndex = newIndex + 1

        new_binary_label_mask = self.binary_label_mask[maskToKeep]
        new_pat_label_mask = self.patNum_label_mask[maskToKeep]
        new_time_label_mask = self.time_label_mask[maskToKeep, :]
        new_move_label_mask = self.move_label_mask[maskToKeep]
        new_speed_label_mask = self.speed_label_mask[maskToKeep]
        new_dir_label_mask = self.dir_label_mask[maskToKeep]

        self._examples = filt_examples
        self._num_examples = new_numExamples
        self._data_files = new_dataFiles
        self.binary_label_mask = new_binary_label_mask
        self.patNum_label_mask = new_pat_label_mask
        self.time_label_mask = new_time_label_mask
        self.move_label_mask = new_move_label_mask
        self.speed_label_mask = new_speed_label_mask
        self.dir_label_mask = new_dir_label_mask

        print('Data filtered out for direction classification, %d examples, %d are positive %d are negative' %(new_numExamples, numPosExamps,numNegExamps))



    def setDataForSpeedClassification(self):
        #maskToKeep = np.ones(self._num_examples, dtype=bool)
        #maskToKeep[:] = False

        print('Filtering dataset for Speed classification only')

        maskToKeep = (self.move_label_mask == 1) & (self.speed_label_mask >= 1)

        filt_examples = {}
        new_numExamples = np.sum(maskToKeep)
        new_dataFiles = []


        sampsIndToInclude = np.nonzero(maskToKeep == True)

        newIndex = 0

        numPosExamps = 0
        numNegExamps = 0

        for kInd in sampsIndToInclude[0]:
            item = list(self._examples[kInd])
            if self.speed_label_mask[kInd] == 1:
                item[1] = np.zeros(1, dtype=np.float64)
                numNegExamps = numNegExamps + 1
            elif self.speed_label_mask[kInd] == 2:
                item[1] = np.ones(1, dtype=np.float64)
                numPosExamps = numPosExamps + 1

            self.binary_label_mask[kInd] = item[1] > 0
            filt_examples[newIndex] = tuple(item)
            new_dataFiles.append(self._data_files[kInd])
            newIndex = newIndex + 1

        new_binary_label_mask = self.binary_label_mask[maskToKeep]
        new_pat_label_mask = self.patNum_label_mask[maskToKeep]
        new_time_label_mask = self.time_label_mask[maskToKeep, :]
        new_move_label_mask = self.move_label_mask[maskToKeep]
        new_speed_label_mask = self.speed_label_mask[maskToKeep]
        new_dir_label_mask = self.dir_label_mask[maskToKeep]

        self._examples = filt_examples
        self._num_examples = new_numExamples
        self._data_files = new_dataFiles
        self.binary_label_mask = new_binary_label_mask
        self.patNum_label_mask = new_pat_label_mask
        self.time_label_mask = new_time_label_mask
        self.move_label_mask = new_move_label_mask
        self.speed_label_mask = new_speed_label_mask
        self.dir_label_mask = new_dir_label_mask

        print('Data filtered out for speed classification, %d examples, %d are positive %d are negative' %(new_numExamples, numPosExamps,numNegExamps))


    def keepDataByPatient(self, percentPatsToKeep=1.0):
        maskToKeep = np.ones(self._num_examples, dtype=bool)
        maskToKeep[:] = False

        uniquePatNums = np.unique(self.patNum_label_mask)

        numPatsToTake = int(len(uniquePatNums) * percentPatsToKeep)

        if numPatsToTake < 1:
            numPatsToTake = 1

        print('Choosing %d patients' % numPatsToTake)

        #np.random.seed(rSeed)

        indexP = np.random.choice(len(uniquePatNums), numPatsToTake, replace=False)

        for i in indexP:
            i = i.item()
            patNum = uniquePatNums[i]
            print("Keeping patient %d in the set" % patNum)
            maskToKeep[self.patNum_label_mask == patNum] = True

        filt_examples = {}
        new_numExamples = np.sum(maskToKeep)
        new_dataFiles = []
        new_binary_label_mask = self.binary_label_mask[maskToKeep]
        new_pat_label_mask = self.patNum_label_mask[maskToKeep]
        new_time_label_mask = self.time_label_mask[maskToKeep, :]
        new_move_label_mask = self.move_label_mask[maskToKeep]
        new_speed_label_mask = self.speed_label_mask[maskToKeep]
        new_dir_label_mask = self.dir_label_mask[maskToKeep]

        sampsIndToInclude = np.nonzero(maskToKeep == True)

        newIndex = 0

        for kInd in sampsIndToInclude[0]:
            filt_examples[newIndex] = self._examples[kInd]
            new_dataFiles.append(self._data_files[kInd])
            newIndex = newIndex + 1

        self._examples = filt_examples
        self._num_examples = new_numExamples
        self._data_files = new_dataFiles
        self.binary_label_mask = new_binary_label_mask
        self.patNum_label_mask = new_pat_label_mask
        self.time_label_mask = new_time_label_mask
        self.move_label_mask = new_move_label_mask
        self.speed_label_mask = new_speed_label_mask
        self.dir_label_mask = new_dir_label_mask



    def keepData(self , percentOfDataToKeep=1.0):

        maskToKeep = np.ones(self._num_examples, dtype=bool)
        maskToKeep[:] = False

        uniquePatNums = np.unique(self.patNum_label_mask)

        #np.random.seed(rSeed)

        for patNum in uniquePatNums:
            patNum = patNum.item()

            posSampsInd = np.nonzero((self.patNum_label_mask == patNum) & (self.binary_label_mask == True) )
            negSampsInd = np.nonzero( (self.patNum_label_mask == patNum) & (self.binary_label_mask == False) )

            numPosSampsToTake = int( len(posSampsInd[0]) * percentOfDataToKeep)
            if numPosSampsToTake < 1:
                numPosSampsToTake = 1

            numNegSampsToTake = int( len(negSampsInd[0]) * percentOfDataToKeep)
            if numNegSampsToTake < 1:
                numNegSampsToTake = 1

            print("\nPat num %d, has %d pos samps and %d neg samps. Keeping %d pos samps and %d neg samps" % (patNum, len(posSampsInd[0]), len(negSampsInd[0]), numPosSampsToTake, numNegSampsToTake))

            indexP = np.random.choice(len(posSampsInd[0]), numPosSampsToTake, replace=False)
            indexN = np.random.choice(len(negSampsInd[0]), numNegSampsToTake, replace=False)

            #posSampsIndToTake = posSampsInd[0][indexP]
            maskToKeep[posSampsInd[0][indexP]] = True
            maskToKeep[negSampsInd[0][indexN]] = True

        filt_examples = {}
        new_numExamples = np.sum(maskToKeep)
        new_dataFiles = []
        new_binary_label_mask = self.binary_label_mask[maskToKeep]
        new_pat_label_mask = self.patNum_label_mask[maskToKeep]
        new_time_label_mask = self.time_label_mask[maskToKeep, :]
        new_move_label_mask = self.move_label_mask[maskToKeep]
        new_speed_label_mask = self.speed_label_mask[maskToKeep]
        new_dir_label_mask = self.dir_label_mask[maskToKeep]

        sampsIndToInclude = np.nonzero( maskToKeep == True)

        newIndex = 0

        for kInd in sampsIndToInclude[0]:
            filt_examples[newIndex] = self._examples[kInd]
            new_dataFiles.append(self._data_files[kInd])
            newIndex = newIndex + 1

        self._examples = filt_examples
        self._num_examples = new_numExamples
        self._data_files = new_dataFiles
        self.binary_label_mask = new_binary_label_mask
        self.patNum_label_mask = new_pat_label_mask
        self.time_label_mask = new_time_label_mask
        self.move_label_mask = new_move_label_mask
        self.speed_label_mask = new_speed_label_mask
        self.dir_label_mask = new_dir_label_mask


        '''
            self._data_files = [x for sub_dir in sub_dirs for x in glob.glob(os.path.join(data_dir, sub_dir, '*'))]

            self._num_examples = len(self._data_files)
    
            # TODO: Programmatically determine this
            self._seq_len = 1
            
            self.hold_in_memory = hold_in_memory
            if self.hold_in_memory:
              self._initialized = False
              self._examples = {}
              self.binary_label_mask = []
              self.patNum_label_mask = []
              self._populate()
        '''










    def getSpeedLabel(self):
        '''
        print('Populating [', end='', flush=True)
        incro = np.floor(self._num_examples / 60)


        speed_label_mask = np.zeros(self._num_examples, dtype=np.ubyte)

        for i in range(0, self._num_examples):
          if i % incro == 0:
            print('.', end='', flush=True)
          item = self.getitemSpeed(i)
          #self._examples[i] = item
          gtSpd = item[2]

          speed_label_mask[i] = gtSpd

        print('] Done\n', flush=True)
        '''

        '''
        if self._condense_outputs == 2:
            print(np.sum(binary_label_mask))
            print(np.sum(np.logical_not(binary_label_mask)), flush=True)
        elif self._condense_outputs == 3 :
            print("0 class count = %d" % (np.sum(binary_label_mask == 0)  )  )
            print("1 class count = %d" % (np.sum(binary_label_mask == 1)  ) )
            print("2 class count = %d" % (np.sum(binary_label_mask == 2)), flush=True)


        self._initialized = True
        self.binary_label_mask = binary_label_mask
        '''
        return self.speed_label_mask

    def getDirLabelMask(self):
        return self.dir_label_mask


    def getTimeLabel(self):
        return self.time_label_mask
        '''
        print('Populating [', end='', flush=True)
        incro = np.floor(self._num_examples / 60)


        time_label_mask = np.zeros((self._num_examples,3), dtype=np.ubyte)

        for i in range(0, self._num_examples):
          if i % incro == 0:
            print('.', end='', flush=True)
          item = self.getItemTime(i)

          for el in range(0,len(item)):
              time_label_mask[i,el] = item[el]

        print('] Done\n', flush=True)



        return time_label_mask
        '''


    def getPatNumLabel(self):

        '''
        print('Populating [', end='', flush=True)
        incro = np.floor(self._num_examples / 60)


        patNum_label_mask = np.zeros((self._num_examples,1), dtype=np.ubyte)

        for i in range(0, self._num_examples):
          if i % incro == 0:
            print('.', end='', flush=True)
          item = self.getItemPatNum(i)


          patNum_label_mask[i] = item

        print('] Done\n', flush=True)

        return patNum_label_mask
        '''

        return self.patNum_label_mask


    
    def _cal_data_statistics(self):
        np.set_printoptions(precision=1)
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': '{:>5.1f}'.format})
        print('\n\n Parse data range stats...\n\n')
        feat_min = np.zeros(9) + 100000000
        feat_max = np.zeros(9) 
        mns = np.zeros((self._num_examples, 9))
        stdvs = np.zeros((self._num_examples, 9))
        for i in range(0, self._num_examples):
          itm = self.__getitem__(i)
          feats = itm[0]
          for j in range(0, 9):
            inner_f = feats[:, j].flatten()
            if np.max(inner_f) > feat_max[j]:
              feat_max[j] = np.max(inner_f)
            if np.min(inner_f) < feat_min[j]:
              feat_min[j] = np.min(inner_f)
            mns[i, j] = np.mean(inner_f)
            stdvs[i, j] = np.std(inner_f)  
          if i % 100 == 0:
            print(feat_min)
            print(feat_max)
            print(np.mean(mns, axis=0))
            print(np.mean(stdvs, axis=0))
            print('\n\n')



    def __len__(self):
        return self._num_examples


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.hold_in_memory and self._initialized:
            return self._examples[idx]

        # Read file from disk
        curr_example_path = self._data_files[idx]
        example = H5Reader.read(curr_example_path, dataset_name=self._H5_EXAMPLE_NAME, max_attempts=3)

        colsToKeep = self.colsToKeep
        numRows = example.shape[0]
        numCols = example.shape[1]

        # If only using certain sensors
        if len(colsToKeep) != 0:
            nExample = np.zeros((numRows, len(colsToKeep), 1))
            newIndex = 0
            for ck in colsToKeep:
                nExample[0:numRows, newIndex, 0] = example[0:numRows, ck, 0]
                newIndex = newIndex + 1
            example = np.copy(nExample)
            numCols = example.shape[1]

        '''
        print('%d rows, %d cols' % (numRows, numCols))
        print('')
        for r in range(0,numRows):
            for c in range(0,numCols):
                print(("%.2f\t" % (example[r][c][0])), end='')
            print()
        '''

        gt = H5Reader.read(curr_example_path, dataset_name=self._H5_GT_NAME, max_attempts=3)

        # if self._condense_outputs == 1:
        #    gt = condense(gt)
        if self._condense_outputs == 2:
            gt = condense_movement(gt)
        elif self._condense_outputs == 3:
            gt = condense_outputs3class(gt)
        elif self._condense_outputs == 5:
            gt = condense_outputs5class(gt)

        if gt == -1:
            sys.stderr.write("Invalid labeling %s\n" % (curr_example_path))

        if self.norm_feats:
            # Normalization is a simple range reduction to [-1, 1] at the moment.
            # Scratch that, trying [0, 1]:
            # Scratch that, trying [-, 0.5]:
            example = (example - self.norm_min) / (self.norm_max - self.norm_min) - 0.5
            # example = example / self.norm_max

        return example, gt


    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.hold_in_memory and self._initialized:
          return self._examples[idx]
        
        # Read file from disk
        curr_example_path = self._data_files[idx]
        example = H5Reader.read(curr_example_path, dataset_name=self._H5_EXAMPLE_NAME, max_attempts=3)

        colsToKeep = self.colsToKeep
        numRows = example.shape[0]
        numCols = example.shape[1]

        #If only using certain sensors
        if len(colsToKeep) != 0:
            nExample = np.zeros( (numRows,len(colsToKeep),1))
            newIndex = 0
            for ck in colsToKeep:
                nExample[0:numRows,newIndex,0] = example[0:numRows,ck,0]
                newIndex = newIndex + 1
            example = np.copy(nExample)
            numCols = example.shape[1]



        '''
        print('%d rows, %d cols' % (numRows, numCols))
        print('')
        for r in range(0,numRows):
            for c in range(0,numCols):
                print(("%.2f\t" % (example[r][c][0])), end='')
            print()
        '''
            



        gt = H5Reader.read(curr_example_path, dataset_name=self._H5_GT_NAME, max_attempts=3)

        #if self._condense_outputs == 1:
        #    gt = condense(gt)
        if self._condense_outputs == 2:
            gt = condense_movement(gt)
        elif self._condense_outputs == 3:
            gt = condense_outputs3class(gt)
        elif self._condense_outputs == 5:
            gt = condense_outputs5class(gt)

        if gt == -1:
            sys.stderr.write("Invalid labeling %s\n" % (curr_example_path))


        if self.norm_feats:
          # Normalization is a simple range reduction to [-1, 1] at the moment.
          # Scratch that, trying [0, 1]:
          # Scratch that, trying [-, 0.5]:
          example = (example - self.norm_min) / (self.norm_max - self.norm_min) - 0.5
          #example = example / self.norm_max

        return example, gt

    def getitemSpeed(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #if self.hold_in_memory and self._initialized:
        #    return self._examples[idx]

        # Read file from disk
        curr_example_path = self._data_files[idx]
        example = H5Reader.read(curr_example_path, dataset_name=self._H5_EXAMPLE_NAME, max_attempts=3)
        gt = H5Reader.read(curr_example_path, dataset_name=self._H5_GT_NAME, max_attempts=3)

        gtSpd = gt[1]
        # if self._condense_outputs == 1:
        #    gt = condense(gt)
        if self._condense_outputs == 2:
            gt = condense_movement(gt)
        elif self._condense_outputs == 3:
            gt = condense_outputs3class(gt)

        if gt == -1:
            sys.stderr.write("Invalid labeling %s\n" % (curr_example_path))

        if self.norm_feats:
            # Normalization is a simple range reduction to [-1, 1] at the moment.
            # Scratch that, trying [0, 1]:
            # Scratch that, trying [-, 0.5]:
            example = (example - self.norm_min) / (self.norm_max - self.norm_min) - 0.5
            # example = example / self.norm_max

        return example, gt, gtSpd

    def getItemTime(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #if self.hold_in_memory and self._initialized:
        #    return self._examples[idx]

        # Read file from disk
        curr_example_path = self._data_files[idx]
        #example = H5Reader.read(curr_example_path, dataset_name=self._H5_EXAMPLE_NAME, max_attempts=3)
        #gt = H5Reader.read(curr_example_path, dataset_name=self._H5_GT_NAME, max_attempts=3)
        timeLabel = H5Reader.read(curr_example_path, dataset_name='time', max_attempts=3)



        return timeLabel

    def getItemGTVec(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #if self.hold_in_memory and self._initialized:
        #    return self._examples[idx]

        # Read file from disk
        curr_example_path = self._data_files[idx]
        #example = H5Reader.read(curr_example_path, dataset_name=self._H5_EXAMPLE_NAME, max_attempts=3)
        #gt = H5Reader.read(curr_example_path, dataset_name=self._H5_GT_NAME, max_attempts=3)
        gtVec = H5Reader.read(curr_example_path, dataset_name=self._H5_GT_NAME, max_attempts=3)

        return gtVec


    def getItemPatNum(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #if self.hold_in_memory and self._initialized:
        #    return self._examples[idx]

        # Read file from disk
        curr_example_path = self._data_files[idx]
        #example = H5Reader.read(curr_example_path, dataset_name=self._H5_EXAMPLE_NAME, max_attempts=3)
        #gt = H5Reader.read(curr_example_path, dataset_name=self._H5_GT_NAME, max_attempts=3)
        patientNum = H5Reader.read(curr_example_path, dataset_name='patientNum', max_attempts=3)



        return patientNum



    # This is currently based on movement class and is only tested to work when condensed_output = 2
    # An example with more timesteps with a higher weighted class has a higher weight overall (the sum of its parts)
    def get_example_weights(self):
        example_weights = np.zeros(len(self))
        class_weights = self.get_class_weights()
        # Iterate over each example
        for i in range(len(self)):
            _, gt = self[i]
            curr_weight = 0
            # Iterate over each timestep
            for curr_gt in gt:
                curr_weight += class_weights[int(curr_gt)]
            example_weights[i] = curr_weight
        return example_weights

    # Only looks at first output (binary eye movement)
    # pow_val (optional) raises the weights to the power of pow_val
    def get_class_weights(self, pow_val=1):
        if self._condense_outputs:
            return self._get_condensed_class_weights(pow_val)
        num_classes = 2
        movement_labels = [int(x) for _, gt in self for x in gt[0]]
        class_freq = [0] * num_classes
        for item in movement_labels:
            class_freq[item] += 1
        class_weights = [1 - (x / (self._num_examples * self._seq_len)) for x in class_freq]
        return [pow(x, pow_val) for x in class_weights]


    # Only tested with condensed_outputs = 1
    def _get_condensed_class_weights(self, pow_val=1):
        num_classes = 5 if self._condense_outputs == 1 else 2
        labels = [x for _, gt in self for x in gt]
        class_freq = [0] * num_classes
        for item in labels:
            class_freq[int(item)] += 1
        raw_weights = [1 - (x / (self._num_examples * self._seq_len)) for x in class_freq]
        weights_sum = sum(raw_weights)
        class_weights = [x / weights_sum for x in raw_weights]
        return [pow(x, pow_val) for x in class_weights]


    # Throw out invalid combinations of movement/speed/direction
    # Example: You can't have a nonzero direction or speed when movement is 0
    def _clean_data(self):
        if self._condense_outputs < 3:
            print("Warning: Currently can only clean when condensing outputs!")
            return
        examples_to_remove = []
        for i in range(len(self)):
            _, label = self[i]
            if not all(label + 1):
                examples_to_remove.append(i)

        print("Data cleaning: Removing %d examples for invalid combinations of outputs" % len(examples_to_remove))
        # Traverse in reverse so popping doesn't shift indices
        for i in reversed(examples_to_remove):
            self._data_files.pop(i)

        # Update number of examples to reflect changes
        self._num_examples = len(self._data_files)
        print("Num examples %d. Length of examples %d" % (self._num_examples, len(self._examples) ) )
        print()


    def writeTSfile(self, num_classes, num_feats, outputFileN, comment):
        fileOut = open(outputFileN, "w")

        fileOut.write("#%s\n\n" %( comment))
        fileOut.write("@problemName EOG\n")
        fileOut.write("@timeStamps false\n")


        if num_feats == 1:
            fileOut.write("@univariate true\n")
        else:
            fileOut.write("@univariate false\n")

        fileOut.write("@classLabel true")
        for c in range(0,num_classes):
            fileOut.write(" %d" % (c))
        fileOut.write("\n\n")
        fileOut.write("@data\n")

        exampNum = 0
        #print(len(self))

        for i in range(len(self)): #for ex in self:
            vals, gt = self[i]
            #gt = ex[1]

            #vals = ex[0]
            numTimeStamps = len(vals)
            numFeatures = len(vals[0])

            # for ts in range(0,len(vals)):
            # feat = vals[ts]
            # numFeatures = len(feat)
            for fi in range(0, numFeatures):
                for ts in range(0, numTimeStamps):
                    #print("Time %d feat %d = %f" % (ts, fi, vals[ts][fi]))
                    fileOut.write("%f" %(vals[ts][fi]))
                    if ts == numTimeStamps-1:
                        fileOut.write(":")
                    else:
                        fileOut.write(",")

            fileOut.write("%d\n" % (int(gt[0]) ))


        fileOut.close()




