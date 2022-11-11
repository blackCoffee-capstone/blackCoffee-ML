import pandas as pd
import langid
import re

class PreProcessor():
    def __init__(
        self
    ):
        return


    def __call__(
        self,
        input_string
    ):
        return self.preProcess(input_string)


    def preProcess(
        self,
        input_string
    ):  
        if self._isNonKorean(input_string) and self._isNonEnglish(input_string):
            return ''
        
        output_string = self._removeEmoji(input_string)
        output_string = self._addSpaceAfterPunctuation(output_string)
        output_string = self._addSpaceBeforeHashtag(output_string)
        output_string = self._removeDoubleSpace(output_string)

        return output_string
    

    def _isNonKorean(
        self,
        input_string
    ):  
        try: 
            detected_languages = langid.rank(input_string)
        except:
            return True
        
        if ('ko' == detected_languages[0][0]):
            return False
        else :
            return True
        
        

    def _isNonEnglish(
        self,
        input_string
    ):  
        try: 
            detected_languages = langid.rank(input_string)
        except:
            return True

        if ('en' == detected_languages[0][0]):
            return False
        else :
            return True


    def _removeEmoji(
        self,
        input_string
    ):
        
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags = re.UNICODE)
        
        output_string = regrex_pattern.sub(r'',input_string)

        return output_string


    def _addSpaceAfterPunctuation(
        self,
        input_string
    ):  
        output_string =re.sub(r'([.,>)}!?])',r'\1 ',input_string)
        return output_string

    def _addSpaceBeforeHashtag(
        self,
        input_string
    ):
        output_string =re.sub(r'(#+)',r' \1',input_string)
        return output_string


    def _removeDoubleSpace(
        self,
        input_string
    ):
        output_string = re.sub("\s\s+" , " ", input_string)
        return output_string


def testPreProcessor():
    dfTestData = pd.read_excel("testingData\preProcessingTextTestSet.xlsx")
    
    testData = [{'input':x, 'output':y} for x, y in zip(dfTestData['raw'], dfTestData['processed'])]
    preProcessor = PreProcessor()

    errors = []

    for pair in testData:
        if not preProcessor(pair['input']) == pair['output'] :
            errors.append('Failed')
            print('------------\nExpected:'+pair['output']+'\nBut got:'+preProcessor(pair['input'])+'\n------------')
        else:
            errors.append(None)

    assert not errors