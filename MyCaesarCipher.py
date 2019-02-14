import pyperclip # optional

class MyCaesarCipher:
    
    
    
    def __init__(self, message: str, mode: str, key: int):
        self.message = message
        self.mode = mode
        self.key = key
        
    def encrytdecrypt(self) -> str:
        # scope 처리 향후 더하기
        SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 !?.'
        idx: int
        returnMsg = ''
        
        
        for symbol in self.message:
            if symbol in SYMBOLS:
                index = SYMBOLS.find(symbol)
                if self.mode == 'encrypt':
                    mutateIdx = idx + self.key
                elif self.mode == 'decrypt':
                    mutateIdx = idx - self.key
                    
                # produced index be in the range     
                if mutateIdx >= len(SYMBOLS):
                    mutateIdx -= len(SYMBOLS)
                elif mutateIdx < 0:
                    mutateIdx += len(SYMBOLS)
                    
                # append a char to the return
                returnMsg = returnMsg + SYMBOLS[mutateIndex]
                
            else: # strange character
                returnMsg = returnMsg + symbol
        
        return returnMsg

    
test1 = MyCaesarCipher('hello, #world@!','encrypt',16)
pyperclip.copy(test1.encrytdecrypt())
print(test1.encrytdecrypt())

test2 = MyCaesarCipher(pyperclip.paste(),'decrypt',16)
print(test2.encrytdecrypt())
