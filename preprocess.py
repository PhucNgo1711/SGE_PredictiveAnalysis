import numpy as np

class Preprocess: 
    rawData = None
    qconf = None

    # resourceIndx = [13, 36, 37, 38, 42, 39]
    resourceIndx = [13, 36, 37, 38, 42]
    filteredData = [] # actual dataset

    # TODO populate programmatically for other flags
    hashmap = { '-q' : {'short' : 0,
                        'all.q' : 0 },
                '-pe' : {'thread' : 1}} 

    def readData(self, rawDataFile, qconfFile):
        # ../AccountingRecords/accounting.last15
        self.rawData = np.genfromtxt(fname=rawDataFile, delimiter=":", dtype="str", max_rows=5)
        # print (rawData)
                
        self.qconf = np.genfromtxt(fname=qconfFile, dtype="str", skip_header=2, skip_footer=1)

        return self.rawData

    def standardizeVal(self, val):
        try:
            int(val) # check if number
        except:
            # not number so NONE, TRUE/FALSE, or time
            if val == "NONE" or val == "FALSE":
                val = 0
            elif val == "TRUE":
                val = 1
            # elif ':' in val:
            #     val = 9999 # TODO standardize time val
            else: # other strings
                if '-' in val: # TODO range request
                    print ("HIT RANGE")
                # elif '.' not in val: # memory storage parsing
                else: # memory storage parsing
                    memType = val[len(val) - 1]
                    val = val[:-1]

                    try:
                        val = int(val)
                    except:
                        val = float(val)

                    if memType == 'G':
                        val = int(val) * 1000
                    elif memType == 'k':
                        val = int(val) / 1000
                # else:
                #     val = 9999 # TODO parse host name

        return val

    def setupHash(self):
        # populate hashmap from reading from qconf output file
        hm = {} # sub hashmap of param name and default val
        for param in self.qconf:
            # p = param.split()
            # check requestable column
            requestable = True if param[4] == "YES" else False 

            if requestable and param[2] != "RESTRING" and param[2] != "TIME" and param[2] != "HOST":
                # print(param[0])
                stdVal = self.standardizeVal(param[6])

                # if stdVal != '#': # TODO for now skip host name and time 0:0:0
                hm[param[0]] = stdVal

        self.hashmap['-l'] = hm

        # print (hashmap)

    def preprocessData(self):
        self.setupHash()

        # TODO 
        # ignore in accurate data lines

        # for each accounting record
        for line in self.rawData:
            # add ru_wallclock, cpu, mem, io, maxvmem first
            appendList = []
            appendList.extend(line[self.resourceIndx])

            # split category field by each flag - (separated by space)
            categories = line[39].split()
            
            # go through each flag and its params
            for i in range(0, len(categories)):        
                # if found flag name
                # if categories[i][0] == '-': 
                if categories[i] == '-l': 
                    key = categories[i] # hash key is flag name
                    val = self.hashmap[key]

                    i = i + 1

                    # split by , for -l
                    param = categories[i].split(",")
                    for p in param:
                        nameVal = p.split("=") # split by = for -l
                        if len(nameVal) > 1:
                            # if stdVal != '#': # TODO for now skip host name and time 0:0:0
                            if nameVal[0] in val:
                                stdVal = self.standardizeVal(nameVal[1])
                        
                                val[nameVal[0]] = stdVal # replace default val

                        # TODO else: # for other flags
                            # split by space for -pe keep the same for -q
                    
                    # append all values of keys to filteredData 
                    # if value of key is not updated in previous loop then there is already default val
                    for paramKey in val:
                        appendList.append(val[paramKey])

            self.filteredData.append(appendList)
        
        return self.filteredData
        