import sys, os, traceback
# import spacy
# from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
# from nltk.tokenize import word_tokenize
# from numpy import unique
# import nltk
# import json
import subprocess


def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.makedirs(fopOutput, exist_ok=True)
        #print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")


# def initDefaultTextEnvi():
#     nlp_model = spacy.load('en_core_web_sm')
#     nlp = English()
#     nlp.add_pipe(nlp.create_pipe('sentencizer'))
#     return nlp_model,nlp
#
# def getSentences(text,nlp):
#     result=None
#     try:
#         document = nlp(text)
#         result= [sent.string.strip() for sent in document.sents]
#     except Exception as e:
#         print('sone error occured {}'.format(str(e)))
#     return result
#
#
# def preprocess(textInLine):
#     text = textInLine.lower()
#     doc = word_tokenize(text)
#     # doc = [word for word in doc if word in words]
#     # doc = [word for word in doc if word.isalpha()]
#     return ' '.join(doc)
#
# # Python program to illustrate the intersection
# # of two lists in most simple way
# def intersection(lst1, lst2):
#     lst3 = [value for value in lst1 if value in lst2]
#     return lst3
# def diff(li1, li2):
#     li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
#     return li_dif
#
# def getPOSInfo(strContent):
#     tokens=word_tokenize(strContent)
#     poses=nltk.pos_tag(tokens)
#     return poses
#
#
# def writeDictToFileText(dictParam,fpFile):
#     lstStr=[]
#     for key in dictParam.keys():
#         strItem=key+'\t'+str(dictParam[key])
#         lstStr.append(strItem)
#
#     strContent='\n'.join(lstStr)
#     fFile=open(fpFile,'w')
#     fFile.write(strContent)
#     fFile.close()
# def writeDictToFile(dictNum,dictText,fpNum,fpText):
#     lstStr=[]
#     lstStr2 = []
#     for key in dictNum.keys():
#         strItem=key+'\t'+str(dictNum[key])
#         strList=sorted(unique(dictText[key]))
#         strItem2=key+'\t'+','.join(strList)
#
#         lstStr.append(strItem)
#         lstStr2.append(strItem2)
#     strContent='\n'.join(lstStr)
#     fFile=open(fpNum,'w')
#     fFile.write(strContent)
#     fFile.close()
#     strContent = '\n'.join(lstStr2)
#     fFile = open(fpText, 'w')
#     fFile.write(strContent)
#     fFile.close()
#
# def ngrams(input, n):
#     input = input.split(' ')
#     output = []
#     for i in range(len(input)-n+1):
#         output.append(input[i:i+n])
#     return output
#
# def runASTGenAndSeeResult(fpCode,fpJSon,numOmit):
#     jsonObject=None
#     strCommand = "clang++-11 -Xclang -ast-dump=json " + fpCode + " | sed -n '/XX_MARKER_XX/,$p' > " + fpJSon
#     try:
#
#         stream = os.popen("clang++-11 -Xclang -ast-dump=json "+fpCode+" | sed -n '/XX_MARKER_XX/,$p' > "+fpJSon)
#         output=stream.read()
#         # print(output)
#         f1=open(fpJSon,'r')
#         strJson=f1.read().strip()
#         f1.close()
#         arrJson=strJson.split('\n')
#         lstStr=[]
#         lstStr.append('{\n\t"kind": "root",\n\t"inner": [')
#         for i in range(numOmit,len(arrJson)):
#             lstStr.append(arrJson[i])
#         strResult='\n'.join(lstStr)
#         f1 = open(fpJSon, 'w')
#         f1.write(strResult)
#         f1.close()
#         # print(strResult)
#         jsonObject = json.loads(strResult)
#         # print(strCommand)
#
#     except:
#         strResult = str(sys.exc_info()[0])
#         # print("Exception in user code:")
#         # print("-" * 60)
#         # traceback.print_exc(file=sys.stdout)
#         # print("-" * 60)
#     return jsonObject,strCommand
#
#
# def getGraphDependencyFromText(strText,nlpObj):
#   lstDeps = []
#   lstNodes=[]
#   lstEdges=[]
#   try:
#     output = nlpObj.annotate(strText, properties={
#       'annotators': 'parse',
#       'outputFormat': 'json'
#     })
#     jsonTemp = output
#     strJsonObj = jsonTemp
#     arrSentences=jsonTemp['sentences']
#     dictWords = {}
#     for sentence in arrSentences:
#       jsonDependency = sentence['basicDependencies']
#       for dep in jsonDependency:
#         strDep=dep['dep']
#         source=dep['governorGloss']
#         target=dep['dependentGloss']
#         # itemTuple=(dep['dep'],dep['governorGloss'],dep['dependentGloss'])
#         # lstDeps.append(itemTuple)
#         if source not in dictWords:
#           dictWords[source]=len(dictWords.keys())+1
#           tupleNode=(dictWords[source],'pseudo_node',source)
#           lstNodes.append(tupleNode)
#         if target not in dictWords:
#           dictWords[target]=len(dictWords.keys())+1
#           tupleNode=(dictWords[target],'pseudo_node',target)
#           lstNodes.append(tupleNode)
#         itemTuple=(dictWords[source],dictWords[target],strDep)
#         lstEdges.append(itemTuple)
#   except:
#     strJsonObj = 'Error'
#
#
#   return lstNodes,lstEdges

