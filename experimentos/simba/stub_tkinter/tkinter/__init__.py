# stub vacío para poder importar SimBA sin Tk (no hay GUI)
class _D:
    def __getattr__(self, n): return _D()
    def __call__(self,*a,**k): return _D()
def __getattr__(n):
    return _D if n[0].isupper() else _D()
