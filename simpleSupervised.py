import sys

from preprocess import Preprocess
from clf import Clf
from reg import Reg

prep = Preprocess()

print (sys.argv[1])
rawData = prep.readData(sys.argv[1], sys.argv[2])
filteredData = prep.preprocessData()

# classifier = Clf(rawData, filteredData)
# classifier.splitSet()
# classifier.svmClf()
# classifier.logClf()
# classifier.rfClf()

regression = Reg(rawData, filteredData)
regression.splitSet()
