import rpyc
import MetaTrader5 as mt5

class MT5Service(rpyc.Service):
    """
    Experimental RPyC bridge to allow Mac/Linux Python to talk to 
    Windows MT5 running in Wine.
    """
    def on_connect(self, conn):
        print("Connected to bridge client.")

    def on_disconnect(self, conn):
        print("Disconnected from bridge client.")

    def exposed_initialize(self, login=None, password=None, server=None):
        if login and password and server:
            return mt5.initialize(login=login, password=password, server=server)
        return mt5.initialize()

    def exposed_shutdown(self):
        return mt5.shutdown()

    def exposed_last_error(self):
        return mt5.last_error()

    def exposed_account_info(self):
        acc = mt5.account_info()
        return acc._asdict() if acc else None

    def exposed_copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        # Rates come as a numpy structured array, we return as list of dicts for portability
        if rates is None: return None
        return [dict(zip(rates.dtype.names, x)) for x in rates]

    def exposed_symbol_info_tick(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        return tick._asdict() if tick else None

    def exposed_order_send(self, request):
        return mt5.order_send(request)

    def exposed_get_terminal_info(self):
        return mt5.terminal_info()._asdict()

if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    print("🚀 MT5 Bridge Server starting on port 18812...")
    server = ThreadedServer(MT5Service, port=18812, protocol_config={"allow_public_attrs": True})
    server.start()
