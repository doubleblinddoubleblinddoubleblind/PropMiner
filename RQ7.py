import ast
import json
import traceback

from sklearn.model_selection import train_test_split
from statistics import mean,median
from tree_sitter import Language, Parser
import pickle
from UtilFunctions import *
from os.path import exists
import glob
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
import nltk
from LibForHandleASTTreeSitter import *
nltk.download('wordnet')
englishStemmer=SnowballStemmer("english")
from nltk.stem import WordNetLemmatizer
from pyparsing import OneOrMore, nestedExpr
from pycorenlp import StanfordCoreNLP
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
import fasttext
import pandas as pd
from scipy.spatial import distance

from nltk.translate.bleu_score import sentence_bleu


wordnet_lemmatizer = WordNetLemmatizer()
punctuations="?:!.,;"
strParseResultsType="<class 'pyparsing.ParseResults'>"
strStrType="<class 'str'>"

def tryPreprocessHtmlTags(strInput):
    strOutput=strInput
    try:
        strOutput=BeautifulSoup(strInput).get_text()
    except Exception as e:
        traceback.print_exc()
    return strOutput

def isLeafNode(jsonInput,arrCodes):
    isOK=True
    if 'ci' in jsonInput.keys() and len(jsonInput['ci'])>0:
        isOK=False
    return isOK
def isLeafNodeNLPT(jsonInput):
    isOK=True
    if 'ci' in jsonInput.keys() and len(jsonInput['ci'])>0:
        isOK=False
    return isOK

def increaseNumberToDicts(key,dictItem):
    if not key in dictItem.keys():
        dictItem[key]=1
    else:
        dictItem[key] =dictItem[key]+1
    # print(dictItem)

def parseASTTreeAndKeepTrackLabels(jsonInput,arrCodes,dictItemOutput):
    isLeaf=isLeafNode(jsonInput,arrCodes)
    if not isLeaf:
        if 't' in jsonInput.keys():
            strType=jsonInput['t'].strip()
            if strType!='':
                strLabelInDict='N_T_{}'.format(strType)
                increaseNumberToDicts(strLabelInDict,dictItemOutput)
    else:
        if 't' in jsonInput.keys():
            strType=jsonInput['t'].strip()
            if strType!='':
                strLabelInDict='T_T_{}'.format(strType)
                increaseNumberToDicts(strLabelInDict,dictItemOutput)
        strValue = getTerminalValueFromASTNode(jsonInput,arrCodes)
        if strValue!='':
            if len(strValue)>30:
                strValue=strValue[:30]
            strLabelInDict = 'T_V_{}'.format(strValue)
            increaseNumberToDicts(strLabelInDict, dictItemOutput)
    if 'ci' in jsonInput.keys():
        lstChildren=jsonInput['ci']
        for i in range(0,len(lstChildren)):
            parseASTTreeAndKeepTrackLabels(lstChildren[i],arrCodes,dictItemOutput)

def parseNLPTTreeAndKeepTrackLabels(jsonInput,dictItemOutput):
    isLeaf = isLeafNodeNLPT(jsonInput)
    if not isLeaf:
        if 'ta' in jsonInput.keys():
            strType = jsonInput['ta'].strip()
            if strType != '':
                strLabelInDict = 'N_T_{}'.format(strType)
                increaseNumberToDicts(strLabelInDict, dictItemOutput)
                if strType=='ROOT':
                    # input('go to here ')
                    if 'dep' in jsonInput.keys():
                        lstDeps=jsonInput['dep']
                        # input('go to here ')
                        for i in range(0,len(lstDeps)):
                            tupDep=lstDeps[i]
                            if len(tupDep)>=3:
                                strPropName='E_dep_{}'.format(tupDep[2])
                                increaseNumberToDicts(strPropName,dictItemOutput)
    else:

        if 'ta' in jsonInput.keys():
            strType = jsonInput['ta'].strip()
            if strType != '':
                strLabelInDict = 'T_T_{}'.format(strType)
                increaseNumberToDicts(strLabelInDict, dictItemOutput)

        if 'val' in jsonInput.keys():
            strValue = jsonInput['val'].strip()
            if strValue != '':
                strLabelInDict = 'T_V_{}'.format(strValue)
                increaseNumberToDicts(strLabelInDict, dictItemOutput)
    if 'ci' in jsonInput.keys():
        lstChildren = jsonInput['ci']
        for i in range(0, len(lstChildren)):
            parseNLPTTreeAndKeepTrackLabels(lstChildren[i], dictItemOutput)

def getQuantitiesLabelsForASTs(strItemAST,strCode):
    dictItemOutput={}
    isOK=False
    try:
        arrCodes=strCode.split('\n')
        jsonInput=ast.literal_eval(strItemAST)
        parseASTTreeAndKeepTrackLabels(jsonInput,arrCodes,dictItemOutput)
        if len(dictItemOutput.keys())>0:
            isOK=True

    except Exception as e:
        traceback.print_exc()
        pass
    return dictItemOutput,isOK

def getQuantitiesLabelsForNLPTs(strItemNLPT):
    dictItemOutput={}
    isOK=False
    try:
        jsonInput=ast.literal_eval(strItemNLPT)
        parseNLPTTreeAndKeepTrackLabels(jsonInput,dictItemOutput)
        if len(dictItemOutput.keys())>0:
            isOK=True

    except Exception as e:
        pass
    return dictItemOutput,isOK

def getVectorRepresentationFastext(strInput,model):
    isOK = False
    lstVector=[]
    try:
        tokens = strInput.split()
        strValue=strInput
        if len(tokens) > 500:
            tokens=tokens[:500]
            # print(tokens)
            # input('aaaa')
            strValue=' '.join(strValue)
        lstVector = model.get_sentence_vector(strValue).tolist()
        if len(lstVector)>0:
            isOK=True
    except Exception as e:
        traceback.print_exc()
        print(strInput)
        # input('aaa ')
        pass
    return lstVector,isOK

def addLabelValueToDictLabel(dictLabel, lbl, val):
    # if not 'dict_appears' in dictLabel.keys():
    #     dictLabel['dict_appears']={}
    # dictAppears=dictLabel['dict_appears']
    if not lbl in dictLabel.keys():
        dictLabel[lbl] = {}
    dictValueOfLabels = dictLabel[lbl]
    if not val in dictValueOfLabels.keys():
        dictValueOfLabels[val] = 0
    dictValueOfLabels[val] += 1

def addMeasurementOnDictLabel(dictLabel, totalInstances):
    # dictLabel['total_instances']=totalInstances
    # dictAppears=dictLabel['dict_appears']
    dictMeasurements = {}
    for lbl in dictLabel.keys():
        dictItemValueAndCounts = dictLabel[lbl]
        dictItemValueAndCounts = dict(
            sorted(dictItemValueAndCounts.items(), key=operator.itemgetter(1), reverse=True))
        sumAppearPerLabel = 0
        lstValMinMaxMedium = []
        for val in dictItemValueAndCounts.keys():
            itemCount = dictItemValueAndCounts[val]
            sumAppearPerLabel += itemCount
            for j in range(0, itemCount):
                lstValMinMaxMedium.append(val)

        lenDistinctValues = len(dictItemValueAndCounts.keys())
        avgAppearancePerLabel = sumAppearPerLabel / len(dictItemValueAndCounts.keys())
        lstVals = list(dictItemValueAndCounts.keys())
        minVal = min(lstValMinMaxMedium)
        maxVal = max(lstValMinMaxMedium)
        meanVal = mean(lstValMinMaxMedium)
        medianVal = median(lstValMinMaxMedium)
        percentageAppearance = sumAppearPerLabel / totalInstances
        minAppPercentage = dictItemValueAndCounts[minVal] / sumAppearPerLabel
        maxAppPercentage = dictItemValueAndCounts[maxVal] / sumAppearPerLabel
        dictMeasurements[lbl] = [sumAppearPerLabel, percentageAppearance, avgAppearancePerLabel, lenDistinctValues,
                                 minVal, maxVal, meanVal, medianVal]
        lstColumnNames = ['sumAppearPerLabel', 'percentageAppearance', 'avgAppearancePerLabel', 'lenDistinctValues',
                          'minVal', 'maxVal', 'meanVal', 'medianVal']
    return dictMeasurements, lstColumnNames

def getDataframeFromDict(dictArray,fpEmb,fpLbl):
    lstDimensions1=[]
    lstDimensions2 = []
    for lbl in dictArray.keys():
        lstValAndLabel=dictArray[lbl]
        lstVals=lstValAndLabel[0]
        lstLabels=lstValAndLabel[1]
        lstVals.insert(0,lbl)
        lstLabels.insert(0,lbl)
        lstDimensions1.append(lstVals)
        lstDimensions2.append(lstLabels)

    lstCols1=[]
    for i in range(0,len(lstDimensions1[0])):
        if i==0:
            lstCols1.append('id')
        else:
            lstCols1.append('dim_{}'.format(i+1))

    lstCols2=[]
    for i in range(0,len(lstDimensions2[0])):
        if i==0:
            lstCols2.append('id')
        else:
            lstCols2.append('prop_{}'.format(i+1))

    df1=pd.DataFrame(lstDimensions1,columns=lstCols1)
    df2 = pd.DataFrame(lstDimensions2, columns=lstCols2)
    df1.to_csv(fpEmb,index=False)
    df2.to_csv(fpLbl,index=False)

def performEvaluationOnCodeSearch(dictVectorsOfItemsForEvaluation,dictVectorFasttextForEvaluation,beamsize,fopProjectEvaluationAccuracy):
    #     1. extract list of candidates as code
    # 2. extract vector of input comment
    # 3. compare input comment with list of candidates as code
    listKeys=list(dictVectorsOfItemsForEvaluation.keys())
    halfBeam=beamsize//2
    lstTopKForEachInstances=[]
    lstAllBLEUScore = []
    for i in range(0,len(listKeys)):
        key=listKeys[i]
        startIndex=i-halfBeam
        endIndex=startIndex+beamsize-1
        if startIndex<0:
            startIndex=0
            endIndex=startIndex+beamsize
        if endIndex>=len(listKeys):
            endIndex=len(listKeys)-1
            startIndex=endIndex-beamsize+1

        vectorInputComment=dictVectorsOfItemsForEvaluation[key]['v_gcb_clean_comment']+dictVectorFasttextForEvaluation[key]['v_ft100sk_comment']
        dictScoreAndOutput={}
        for j in range(startIndex,endIndex+1):
            keyCandidate=listKeys[j]
            vectorCandidate=dictVectorsOfItemsForEvaluation[keyCandidate]['v_gcb_code']+dictVectorFasttextForEvaluation[key]['v_ft100sk_code']
            distanceIJ=distance.euclidean(vectorInputComment,vectorCandidate)
            if not distanceIJ in dictScoreAndOutput.keys():
                dictScoreAndOutput[distanceIJ]=set()
            dictScoreAndOutput[distanceIJ].add(j)
        dictScoreAndOutput = {k: dictScoreAndOutput[k] for k in sorted(dictScoreAndOutput)}
        topKForI=100
        # print(lstScoreAndOutput)
        # input('check over')
        lstAllKeyScoreAndCoddeForeachItem = []
        lstKeyScores=list(dictScoreAndOutput.keys())
        for j in range(0,len(lstKeyScores)):
            setIndexesSameScore=dictScoreAndOutput[lstKeyScores[j]]
            if i in setIndexesSameScore:
                topKForI=j+1
                break
        lstTopKForEachInstances.append(topKForI)
        curentRankForCandidate = 0
        topKForLog = 0
        for scoreItem in dictScoreAndOutput.keys():
            lstValueForScore = dictScoreAndOutput[scoreItem]
            for j in lstValueForScore:
                keyCandidate = listKeys[j]
                lstAllKeyScoreAndCoddeForeachItem.append([keyCandidate, topKForLog + 1, scoreItem])
            topKForLog += len(lstValueForScore)
        arrExpectedCode = dictVectorsOfItemsForEvaluation[key]['json']['code'].split()
        arrPredictedCode = dictVectorsOfItemsForEvaluation[lstAllKeyScoreAndCoddeForeachItem[0][0]]['json'][
            'code'].split()
        # print(arrPredictedComment)
        # print(arrExpectedComment)

        bleuScoreItem = sentence_bleu([arrExpectedCode], arrPredictedCode, weights=(0.25, 0.25, 0.25, 0.25))
        # print('bleu {}'.format(bleuScoreItem))
        # input('aaaaaaaaaaaaaaaaaa')
        lstAllBLEUScore.append(bleuScoreItem)
    lstStr=['Top-K,Number,Accuracy']
    for i in range(1,6):
        numberTopI=sum(item<=i for item in lstTopKForEachInstances)
        percentageTopI=numberTopI/len(lstTopKForEachInstances)
        lstStr.append('Top-{},{}/{},{}'.format(i, numberTopI,len(lstTopKForEachInstances), percentageTopI))
    mrrScore = mean([1 / i for i in lstTopKForEachInstances])
    lstStr.append('MRR: {}'.format(mrrScore))
    avgBLEU = mean(lstAllBLEUScore)
    lstStr.append('BLEU: {}'.format(avgBLEU))
    fpAccuracyLog=fopProjectEvaluationAccuracy+'codeSearch_plus2Vectors.txt'
    f1=open(fpAccuracyLog,'w')
    f1.write('\n'.join(lstStr))
    f1.close()

def performEvaluationOnCodeSummarization(dictVectorsOfItemsForEvaluation,dictVectorFasttextForEvaluation, beamsize, fopProjectEvaluationAccuracy):
    #     1. extract list of candidates as code
    # 2. extract vector of input comment
    # 3. compare input comment with list of candidates as code
    listKeys = list(dictVectorsOfItemsForEvaluation.keys())
    halfBeam = beamsize // 2
    lstTopKForEachInstances = []
    lstAllBLEUScore=[]

    for i in range(0, len(listKeys)):
        key = listKeys[i]
        startIndex = i - halfBeam
        endIndex = startIndex + beamsize - 1
        if startIndex < 0:
            startIndex = 0
            endIndex = startIndex + beamsize
        if endIndex >= len(listKeys):
            endIndex = len(listKeys) - 1
            startIndex = endIndex - beamsize + 1

        vectorInputCode = dictVectorsOfItemsForEvaluation[key]['v_gcb_code']+dictVectorFasttextForEvaluation[key]['v_ft100sk_code']
        # print(len(vectorInputCode))
        # input('aaaaaa')
        dictScoreAndOutput = {}
        for j in range(startIndex, endIndex + 1):
            keyCandidate = listKeys[j]
            vectorCandidate = dictVectorsOfItemsForEvaluation[keyCandidate]['v_gcb_clean_comment']+dictVectorFasttextForEvaluation[key]['v_ft100sk_comment']
            distanceIJ = distance.euclidean(vectorInputCode, vectorCandidate)
            if not distanceIJ in dictScoreAndOutput.keys():
                dictScoreAndOutput[distanceIJ] = set()
            dictScoreAndOutput[distanceIJ].add(j)
        dictScoreAndOutput = {k: dictScoreAndOutput[k] for k in sorted(dictScoreAndOutput)}
        topKForI = 100
        lstKeyScores = list(dictScoreAndOutput.keys())
        for j in range(0, len(lstKeyScores)):
            setIndexesSameScore = dictScoreAndOutput[lstKeyScores[j]]
            if i in setIndexesSameScore:
                topKForI = j + 1
                break
        lstTopKForEachInstances.append(topKForI)
        curentRankForCandidate = 0
        topKForLog = 0
        lstAllKeyScoreAndCoddeForeachItem = []
        for scoreItem in dictScoreAndOutput.keys():
            lstValueForScore = dictScoreAndOutput[scoreItem]
            for j in lstValueForScore:
                keyCandidate = listKeys[j]
                lstAllKeyScoreAndCoddeForeachItem.append([keyCandidate, topKForLog + 1, scoreItem])
            topKForLog += len(lstValueForScore)
        arrExpectedComment = dictVectorsOfItemsForEvaluation[key]['json']['comment'].split()
        arrPredictedComment = dictVectorsOfItemsForEvaluation[lstAllKeyScoreAndCoddeForeachItem[0][0]]['json'][
            'comment'].split()
        # print(arrPredictedComment)
        # print(arrExpectedComment)

        bleuScoreItem = sentence_bleu([arrExpectedComment], arrPredictedComment, weights=(0.25, 0.25, 0.25, 0.25))
        # print('bleu {}'.format(bleuScoreItem))
        # input('aaaaaaaaaaaaaaaaaa')
        lstAllBLEUScore.append(bleuScoreItem)
    lstStr = ['Top-K,Number,Accuracy']
    for i in range(1, 6):
        numberTopI = sum(item <= i for item in lstTopKForEachInstances)
        percentageTopI = numberTopI / len(lstTopKForEachInstances)
        lstStr.append('Top-{},{}/{},{}'.format(i, numberTopI,len(lstTopKForEachInstances), percentageTopI))
    mrrScore = mean([1 / i for i in lstTopKForEachInstances])
    lstStr.append('MRR: {}'.format(mrrScore))
    avgBLEU = mean(lstAllBLEUScore)
    lstStr.append('BLEU: {}'.format(avgBLEU))
    fpAccuracyLog = fopProjectEvaluationAccuracy + 'codeSummarization_plus2Vectors.txt'
    f1 = open(fpAccuracyLog, 'w')
    f1.write('\n'.join(lstStr))
    f1.close()


fopHeteroCPData='../../HeteroCP/data/'
fopHeteroCPExp='../../HeteroCP/experiments/'
fopCleanData='../../HeteroCP/data/FSE_2022_CAT_dataset/dataset/clean/'

# input('check ast here')
listOfProjects=['tlcodesum']
# fopCleanCommentNLParseTree= fopHeteroCPData + 'allSPTOfComment/clean/'
# fopRawCommentNLParseTree= fopHeteroCPData + 'allSPTOfComment/raw/'
# fopASTParseTree= fopHeteroCPData + 'astInfos_v1/'
fopEvaluationAccuracy= fopHeteroCPExp + 'RQ9/eval_plus2Vectors/'
fopNLPTNewDimensionResult= fopHeteroCPExp + 'RQ1/fasttext_100_codesearch/result_cose/'
fopASTNewDimensionResult= fopHeteroCPExp + 'RQ1/fasttext_100_codesum/result_cosum/'
createDirIfNotExist(fopEvaluationAccuracy)
fopDictLabelASTAndParseTree=fopHeteroCPData+'dict_labels_spt_nlpt/'
fopDictFasttextBowEmb=fopHeteroCPData+'dict_fasttext_bow_emb/'
# fopNewDimensionResultfopNewDimensionResult=fopHeteroCPExp+'/'
# fopLabelToExcelData= fopHeteroCPData + 'labelToExcels/'
createDirIfNotExist(fopEvaluationAccuracy)
# fpLogCurrentProcess= fopEvaluationAccuracy + 'log_currentProcess_parseTree.txt'
# fpLogASTGenProcess= fopEvaluationAccuracy + 'log_parseTree.txt'


currentProjectIndex=-1
currentFileIndex=0
curIndexInsideFile=-1
currentLanguageParser=None

withoutErrorCases=0
totalCases=0
batchSizeOutput=500
# numParseOK=0
isAbleToRun=False
strEL='_EL_'
beamsize=10
for i in range(0,len(listOfProjects)):
    # if (not isAbleToRun) and i<currentProjectIndex:
    #     continue
    lstFpFile=glob.glob(fopCleanData+listOfProjects[i]+'/**/*.*',recursive=True)
    # input('aaaa ')
    # dictProjectAllInstanceAndLabelsAndVectors={}
    fopProjectEvaluationAccuracy=fopEvaluationAccuracy+listOfProjects[i]+'/'

    createDirIfNotExist(fopProjectEvaluationAccuracy)
    try:
        for j in range(0,len(lstFpFile)):
            # if (not isAbleToRun) and j < currentFileIndex:
            #     continue
            try:
                f1=open(lstFpFile[j],'r')
                strJson=f1.read().strip()
                arrJsons=strJson.split('\n')
                f1.close()
                print('{}\t{}\t{}'.format(listOfProjects[i], lstFpFile[j],len(arrJsons)))
                # jsonPrograms=json.loads(strJson)
                # jsonPrograms=
                # listKeys=list(jsonPrograms.keys())
                subFolderData=lstFpFile[j].replace(fopCleanData,'/')
                fnNameOfFile=os.path.basename(lstFpFile[j])
                if not 'test' in fnNameOfFile:
                    continue
                fopSubFolder=subFolderData.replace(fnNameOfFile,'')
                fopItemLabelAndVectorPkl= (fopDictLabelASTAndParseTree + fopSubFolder + '/' + fnNameOfFile + '/').replace('//', '/')
                fopItemFasttextPkl = (
                            fopDictFasttextBowEmb + fopSubFolder + '/' + fnNameOfFile + '/').replace('//', '/')
                # createDirIfNotExist(fopItemLabelAndVectorPkl)
                currentLabelPklIndex = -1
                dictBatchLabels={}
                currentFasttextPklIndex = -1
                dictFasttextLabels = {}
                dictVectorsOfItemsForEvaluation={}
                dictVectorsOfFasttextEmbForEvaluation={}
                for k in range(0,len(arrJsons)):
                    # if (not isAbleToRun) and k<curIndexInsideFile:
                    #     continue
                    try:
                        isAbleToRun=True
                        totalCases+=1
                        jsonItem = ast.literal_eval(arrJsons[k])
                        newLabelPklIndex=k//batchSizeOutput
                        newFasttextPklIndex=k//batchSizeOutput
                        if newLabelPklIndex!=currentLabelPklIndex:
                            fpItemLabelVectorPkl='{}{}.pkl'.format(fopItemLabelAndVectorPkl,newLabelPklIndex)
                            dictBatchLabels=pickle.load(open(fpItemLabelVectorPkl,'rb'))
                            currentLabelPklIndex=newLabelPklIndex
                        if newFasttextPklIndex!=currentFasttextPklIndex:
                            fpItemFasttextVectorPkl='{}{}.pkl'.format(fopItemFasttextPkl,newLabelPklIndex)
                            dictFasttextLabels=pickle.load(open(fpItemFasttextVectorPkl,'rb'))
                            currentFasttextPklIndex=newFasttextPklIndex
                        if k in dictBatchLabels.keys():
                            dictItemLabelAll=dictBatchLabels[k]
                            dictItemFasttextLabel=dictFasttextLabels[k]
                            dictItemLabelAll['json']=jsonItem

                            if 'l_clean_nlpt' in dictItemLabelAll.keys() and 'v_gcb_clean_comment' in dictItemLabelAll.keys() \
                                    and 'l_ast' in dictItemLabelAll.keys() and 'v_gcb_code' in dictItemLabelAll.keys()\
                                    and 'v_ft100sk_comment' in dictItemFasttextLabel.keys()\
                                    and 'v_ft100sk_code' in dictItemFasttextLabel.keys():
                                dictVectorsOfItemsForEvaluation[k]=dictItemLabelAll
                                dictVectorsOfFasttextEmbForEvaluation[k]=dictItemFasttextLabel
                                # print(dictItemFasttextLabel)
                                # input('aaa')

                    except Exception as e:
                        traceback.print_exc()
                        pass
                    currentTime = datetime.now()
                    strPrint = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(i, j, k, listOfProjects[i], lstFpFile[j],
                                                                         withoutErrorCases, totalCases,
                                                                         currentTime)
                    if totalCases % 100 == 0:
                        print(strPrint.strip())


            except Exception as e:
                traceback.print_exc()
                # f1 = open(fpLogCurrentProcess, 'w')
                # f1.write('{}\t{}\t{}'.format(currentProjectIndex, currentFileIndex, curIndexInsideFile))
                # f1.close()
            print('evaluation result {} {}'.format(len(dictVectorsOfItemsForEvaluation.keys()),len(dictVectorsOfFasttextEmbForEvaluation.keys())))
            performEvaluationOnCodeSearch(dictVectorsOfItemsForEvaluation,dictVectorsOfFasttextEmbForEvaluation,beamsize,fopProjectEvaluationAccuracy)
            performEvaluationOnCodeSummarization(dictVectorsOfItemsForEvaluation,dictVectorsOfFasttextEmbForEvaluation, beamsize, fopProjectEvaluationAccuracy)
            # input('aaa ')
            # break
    except Exception as e:
        traceback.print_exc()
        # f1=open(fpLogCurrentProcess,'w')
        # f1.write('{}\t{}\t{}'.format(currentProjectIndex,currentFileIndex,curIndexInsideFile))
        # f1.close()
