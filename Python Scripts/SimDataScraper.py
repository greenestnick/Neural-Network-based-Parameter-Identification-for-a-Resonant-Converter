import os
import torch
from torch.utils.data import Dataset, random_split
import ltspice
import numpy as np

def EngNotationToFloat(num):
    """
        Returns a string of a number that is reduced with a metric prefix as a character 
    """
    lastChar = num[-1]
    fullNum = num[0:-1]

    match lastChar:
        case 'p':
            return float(fullNum) * 10**(-12)
        case 'n':
            return float(fullNum) * 10**(-9)
        case 'u':
            return float(fullNum) * 10**(-6)
        case 'm':
            return float(fullNum) * 10**(-3)
        case 'K':
            return float(fullNum) * 10**(3)
        case 'M':
            return float(fullNum) * 10**(6)
        case 'G':
            return float(fullNum) * 10**(9)
        case 'T':
            return float(fullNum) * 10**(12)
        case _:
            return float(num)

def _GatherSims(DIR):
    """
        Returns a list of successful SIDs from a directory
    """

    SID_Dict = {}
    SID_Failed_Dict = {}
    dirList = os.listdir(DIR)
    
    for file in dirList:
        fileSplit = file.split('.')
        fileNameSplit = fileSplit[0].split("_")
        isThermFile = (fileNameSplit[0] != "SIM")

        if fileSplit[-1] == "txt":
            continue
        if fileSplit[0][-6:] == "FAILED":
            SID = fileNameSplit[1]
            SID_Failed_Dict.setdefault(SID)
        elif isThermFile:
            SID = fileNameSplit[2]
            SID_Dict.setdefault(SID, []).append(fileNameSplit[1])

    # Sort out any sims with missing thermal sims
    TEMP_DICT = {}
    for SID in SID_Dict:
        if len(SID_Dict[SID]) == 2:
            TEMP_DICT[SID] = SID_Dict[SID]
        else:
            SID_Failed_Dict.setdefault(SID)
    SID_Dict = TEMP_DICT
    del TEMP_DICT

    failureRate = len(SID_Failed_Dict)/(len(SID_Failed_Dict) + len(SID_Dict))
    return list(SID_Dict.keys()), failureRate

def _ScrapeData(SID_Keys, DIR):
    """
        Returns a dictionary of parameters and their values for every SID
        
        ### Inputs:
            - SID_KEYS = List of SIDs
            - DIR = directory where SID files are located
    """

    simDict = {}
	
    loadingCount = 0
    for SID in SID_Keys:
        print("SCRAPING: %d/%d (%s)" % (loadingCount + 1, len(SID_Keys), SID))
        simDict.setdefault("SID", []).append(SID)
        iterDict = {}

        # Read the net file params
        with open(DIR + "\\SIM_" + SID + ".net", 'r') as file:
            line = file.readline()

            while line.strip() != ".end":
                if line.startswith(".param"):
                    lineParams = line.removeprefix(".param ").split()
                    for x in lineParams:
                        tokens = x.split('=', 1)
                        iterDict[tokens[0]] = EngNotationToFloat(tokens[1])

                line = file.readline()

        # Read the log file measurements
        with open(DIR + "\\SIM_" + SID + ".log", 'r') as file:
            line = file.readline()
            while line != "":
                if len(line.split("FROM", 1)) > 1:
                    tokens = line.split(':', 1)
                    if len(tokens) == 1:
                        tokens = tokens[0].split("=", 1)
                        iterDict[tokens[0]] = float(tokens[1].split("FROM", 1)[0])
                        # measDict[tokens[0]] = float(tokens[1].split("FROM", 1)[0]) > 0    # Alternative where ZVS is only a boolean
                    else:
                        iterDict[tokens[0]] = float(tokens[1].split("FROM", 1)[0].split("=",1)[1])
                
                line = file.readline()

        #iterDict |= _GenerateMissingChargeMeasurement(DIR, SID, iterDict["F"])

        # Read the thermal log file measurements
        def _ThermalMeasurementsRead(SID, isTop):
            thermDict = {}
            titlePrefix = "\\THERM_" + ("T" if isTop else "B")

            with open(DIR  + titlePrefix + "_" + SID + ".log" , 'r') as file:
                line = file.readline()
                while line != "":

                    if line.startswith(".step"):
                        thermDict.setdefault("Tc", []).append(line.split()[1].split('=', 1)[1])
                    elif line.startswith("Measurement:"):
                        line = file.readline() # Ignore line right under "Measurement" line as its not important
                        for i in range(len(thermDict["Tc"])):
                            thermDict.setdefault("Tj_Final_" + ("T" if isTop else "B"), []).append(float(file.readline().split()[1]))

                    line = file.readline()

            return thermDict
        
        thermDict = _ThermalMeasurementsRead(SID, True) | _ThermalMeasurementsRead(SID, False)
        for i in range(len(thermDict["Tc"])):
            Tc = float(thermDict["Tc"][i])
            Tjt = float(thermDict["Tj_Final_T"][i])
            Tjb = float(thermDict["Tj_Final_B"][i])
			
            iterDict["Tc"] = Tc
            iterDict["Tj_Final_T"] = Tjt
            iterDict["Tj_Final_B"] = Tjb

            for key in iterDict: #TODO REPEAT SID FOR EACH TC
                simDict.setdefault(key, []).append(iterDict[key])
		
        loadingCount += 1

    return simDict

def _IntegralTrap(time, data):
    """
        Takes the trapazoidal integral over some time/data lists
        
        ### Inputs:
            - time - list containing time stamps to the corresponding data list
            - data - list of data points 
    """
        
    accum = 0
    for i in range(1, len(data)):
        y1 = data[i]
        y0 = data[i - 1]
        minY = min(y0, y1)
        maxY = max(y0, y1)

        y = minY + 0.5 * (maxY - minY)
        delX = time[i] - time[i - 1]
        accum += (y * delX)
    
    return accum

def _WindowDerivative(time, data, winN):
    """
        Takes the discrete derivative over are window
        
        ### Inputs:
            - time - list containing time stamps to the corresponding data list
            - data - list of data points 
            - winN - half the window size the derivative takes its difference over
    """

    diffS = [0] * winN
    for i in range(winN, len(time) - winN):
        diff = data[i + winN] - data[i - winN]
        timeDiff = time[i + winN] - time[i - winN]
        diffS.append(diff / timeDiff)
    diffS.extend([0] * winN)
    return diffS

def _GenerateSlewRate(DIR, SID, freq, measureString, varName):
    """
        Finds the slew rate measurements (min, max, avg, rms, and p2p) of a variable inside the three cycle measurement window
        
        ### Inputs:
            - DIR - directory where the simulation files are
            - SID - simulation ID hash 
            - freq - switching frequency used for that particular simulation. Used to define the measurement window of three cyles
            - measureString - LTSpice parameter term the slew rate will be taken for (ex: V(Vp, A) == "v_cs")
            - varName - desired name for the slew rate variable that will be used as the key in the dataset dictionary
    """
        
    l = ltspice.Ltspice(DIR + "\\SIM_" + SID + ".raw")
    l.parse()
    time = l.get_time()
    
    # Only measuring from the 1st to 4th cycle
    diff = np.array(time - ([1/freq] * len(time)))
    start = np.argmin(np.absolute(diff))
    diff = np.array(time - ([4/freq] * len(time)))
    end = np.argmin(np.absolute(diff))

    slewDict = {}
    sig = l.get_data(measureString)

    deriv = np.array(_WindowDerivative(time, sig, 4))
    avg = _IntegralTrap(time[start:end], deriv[start:end])/ (time[end]-time[start])
    rms = np.sqrt(_IntegralTrap(time[start:end], deriv[start:end]**2)/ (time[end]-time[start]))
        
    slewDict[varName + "_slew_max"] = np.max(deriv)
    slewDict[varName + "_slew_min"] = np.min(deriv)
    slewDict[varName + "_slew_pp"] = max(deriv) - min(deriv)
    slewDict[varName + "_slew_avg"] = avg
    slewDict[varName + "_slew_rms"] = rms

    return slewDict

def _GeneratePhaseDifference(DIR, SID, freq):
    l = ltspice.Ltspice(DIR + "\\SIM_" + SID + ".raw")
    l.parse()
    time = l.get_time()

    vin = l.get_data("V(vp,vn)")
    vout = l.get_data("V(vo,vn)")

    _SearchFloat = lambda X, x: np.argmin(np.absolute(  np.array(  X - ([x] * len(X))  )  ))


    timeDeltas = []
    for i in range(3):
        outCycleStart = 1.05 + i
        outCycleEnd = 1.95 + i

        inCycleStart = 1.450 + i
        inCycleEnd = 1.50005 + i

        outStart = _SearchFloat(time, (outCycleStart/freq))
        outEnd = _SearchFloat(time, (outCycleEnd/freq))
        inStart = _SearchFloat(time, (inCycleStart/freq))
        inEnd = _SearchFloat(time, (inCycleEnd/freq))

        outCrossingIndex = _SearchFloat(vout[outStart:outEnd], 0.0) + outStart
        inCrossingIndex = _SearchFloat(vin[inStart:inEnd], 0.0) + inStart

        timeDiff = time[outCrossingIndex] - time[inCrossingIndex]
        timeDeltas.append(timeDiff * freq * 2 * np.pi)

    phaseDict = {}
    phaseDict["phase_diff_avg"] = np.average(timeDeltas)
    phaseDict["phase_diff_max"] = np.max(timeDeltas)
    phaseDict["phase_diff_min"] = np.min(timeDeltas)
    return phaseDict

class SimulationDataset(Dataset):
    """
        Custom dataset object inheriting from the Pytorch dataset class. 
    """

    def __init__(self, dir : str, inputs : list, outputs : list, transform = None):
        self.dir = dir
        self.inputs = inputs
        self.outputs = outputs 
        self.transform = transform
        
        keys, self.failureRate = _GatherSims(dir)
        self.rawDict = _ScrapeData(keys, dir)
        
        if transform:
            self.SimDict = {}
            for key in  self.rawDict:
                if key != "SID":
                    self.SimDict[key] = transform(self.rawDict[key])
        else:
            self.SimDict =  self.rawDict

    def __getitem__(self, index):
        return self.packed_data[index]
    
    def __len__(self):
        return len(self.RawDict["F"])
    
    def Split(self, train_test_ratio, generator = None):
        """
            Used to split the dataset based on a train/test ratio. Interally, it uses the pytorch random_split method
        """
        return random_split(self, [train_test_ratio, 1.0 - train_test_ratio], generator=generator)
	
    def MergeDataset(self, mergingDataset):
        """
            Used to combine dataset. Given the input dataset, this method will combine that data with the current dataset.
            ### Inputs:
                - mergingDataset - secondary dataset to combine with

            ### Output:
                no return dataset, instead the current dataset absorbs the other dataset
        """
            
        for key in self.rawDict:
            self.rawDict[key] += mergingDataset.rawDict[key]
        self.ReTransform()

    def ReTransform(self):
        """
            Applies the transform to the dataset. Useful for renormalizing a dataset after modifying it. 
        """

        if self.transform:
            self.SimDict = {}
            for key in self.rawDict:
                if key != "SID":
                    self.SimDict[key] = self.transform(self.rawDict[key])
        else:
            self.SimDict = self.rawDict
	
    def CreateSubSet(self, newSize):
        """
            Used to downsample the dataset through random selection given the newSize input
        """
        randVec = np.random.choice(len(self), size=newSize, replace=False)
        for key in self.rawDict:
            self.rawDict[key] = (np.array(self.rawDict[key])[randVec]).tolist()
        for key in self.SimDict:
            self.SimDict[key] = (np.array(self.SimDict[key])[randVec]).tolist()

    def PrePackSims(self):
        """
            To avoid gathering data directly from the internal dictionary structure, this method combines all input/output pairs into one long list which is must faster when training.
        """
        numSims = len(self)
        self.packed_data = []

        for i in range(numSims):
            ins = []   
            outs = []     

            for key in self.inputs:
                ins.append(self.SimDict[key][i])
            for key in self.outputs:
                outs.append(self.SimDict[key][i])

            self.packed_data.append((torch.Tensor(ins),torch.Tensor(outs)))

class MinMaxScale(object):
    """
        Transformation object used when normalizing the dataset's data. The object form makes it useful to pass as an argument when creating the dataset and follows Pytorchs conventions for transformations.
    """

    def __call__(self, paramList : list):
        listMin = min(paramList)
        listMax = max(paramList)

        if listMin == listMax:
            return ([1] * len(paramList))
        
        scaledList = []
        for x in paramList:
            xNew = (x - float(listMin)) / (listMax - listMin)
            scaledList.append(xNew)
    
        return scaledList