import pandas as pd


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
        if self._isNonKorean(input_string):
            if self._isNonEnglish(input_string):
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
        
        if():
            return True

        else:
            return False


    def _isNonEnglish(
        self,
        input_string
    ):  
        
        if():
            return True

        else:
            return False


    def _removeEmoji(
        self,
        input_string
    ):
        output_string = input_string
        return output_string


    def _addSpaceAfterPunctuation(
        self,
        input_string
    ):
        output_string = input_string
        return output_string


    def _addSpaceBeforeHashtag(
        self,
        input_string
    ):
        output_string = input_string
        return output_string


    def _removeDoubleSpace(
        self,
        input_string
    ):
        output_string = input_string
        return output_string


def testPreProcessor():
    dfTestData = pd.read_excel("testingData\preProcessingTextTestSet.xlsx")
    
    testData = [{'input':x, 'output':y} for x, y in zip(dfTestData['raw'], dfTestData['processed'])]
    preProcessor = PreProcessor()

    errors = []

    for pair in testData:
        if not preProcessor(pair['input']) == pair['output'] :
            errors.append('Failed')
        else:
            errors.append(None)

    assert not errors