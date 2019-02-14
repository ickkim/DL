class MyCipherHacker:

    def __init__(self, message: str):
        self.message = message
    
    def MyHack(self) -> list:
        # next time, with smarter scope
        SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 !?.'
        
        WHOLEMSG = list()
    
        for key in range(len(SYMBOLS)):
            translatedMsg = ''
            for symbol in self.message:
                if symbol in SYMBOLS:
                    symbolIndex = SYMBOLS.find(symbol)
                    translatedIndex = symbolIndex - key
 
                    # Handle the wrap-around:
                    if translatedIndex < 0:
                        translatedIndex += len(SYMBOLS)

                    # Append the decrypted symbol:
                    translatedMsg = translatedMsg +  SYMBOLS[translatedIndex]

                else:
                    # Append the symbol without encrypting/decrypting:
                    translatedMsg = translated + symbol
            # Display every possible decryption:
            #print('Key #%s: %s' % (key, translatedMsg))
            WHOLEMSG.append(translatedMsg)
        return WHOLEMSG
    
    
# !@#$hello, world%^*.
# R@#$2y669,QC9!6x%^*T

# I love you so much.
# cQ69ByQE9AQ?9Q7Aw2T

# guv6Jv6Jz!J6rp5r7Jzr66ntrM
# This is my secret message.

test1 = MyCipherHacker('guv6Jv6Jz!J6rp5r7Jzr66ntrM')
