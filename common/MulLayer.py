class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    def backward(self, dout):
        dx = dout * dy
        dy = dout * dx
        
        return dx, dy
