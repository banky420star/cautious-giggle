class HybridBrain:
    def __init__(self, risk, executor):
        self.risk = risk
        self.executor = executor

    def live_trade(self, symbol, exposure, max_lots):
        if not self.risk.can_trade():
            return
        self.executor.reconcile_exposure(symbol, exposure, max_lots)
