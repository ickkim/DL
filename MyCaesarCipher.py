class MyCaesarCipher:
    
    SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 !?.'
    returnMsg = ''
    
    def __init__(self, message: str, mode: str, key: int):
        self.message = message
        self.mode = mode
        self.key = key
        
    def encrytdecrypt(self) -> str:
        for symbol in self.message:
            if symbol in SYMBOLS:
                index = SYMBOLS.find(symbol)
                if self.mode == 'encrypt':
                    mutateIndex = idex + self.key
                elif self.mode == 'decrypt':
                    mutateIndex = idex - self.key
                    
                # produced index be in the range     
                if mutateIndex >= len(SYMBOLS):
                    #mutateIndex -= len(SYMBOLS)
                    mutateIndex = mutateIndex - len(SYMBOLS)
                elif mutateIndex < 0:
                    #mutateIndex += len(SYMBOLS)
                    mutateIndex = mutateIndex + len(SYMBOLS)
                
                # append a char to the return
                returnMsg = returnMsg + SYMBOLS[mutateIndex]
                
            else: # strange character
                returnMsg = returnMsg + symbol
        
        return returnMsg
        print(returnMsg)
        
