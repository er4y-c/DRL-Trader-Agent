import pandas as pd

class SupportResistanceDetector:
    """
    This class helps to implementing London breakout strategy by detecting support and resistance levels on sessions.
    """
    def __init__(self):
        self.support = None
        self.resistance = None
        self.stop = None

    def detect(self, state):
        breakout_signal = 0.0
        # Asya oturumu ise
        if state[0][14] == 1: # 14. sütun session sütunu
            if self.support is None or state[0][3] < self.support: # 3. index close sütunu
                self.support = state[0][3]  # Destek seviyesini güncelle / 3. index close sütunu
            if self.resistance is None or state[0][1] > self.resistance: # 1. index high sütunu
                self.resistance = state[0][1]  # Direnç seviyesini güncelle
        # Londra open oturumu ise
        elif state[0][14] == 2:
            # Eğer Londra open oturumunda destek seviyesinin altında kapanış yapılırsa
            if self.support is not None and state[0][3] < self.support:
                breakout_signal = 1  # Destek kırıldı
            # Eğer Londra open oturumunda direnç seviyesine ulaşılırsa
            elif self.support is not None and state[0][3] >= self.resistance:
                breakout_signal = 2  # Direnç kırıldı
     
        return breakout_signal
    
    def reset(self):
        self.support = None
        self.resistance = None