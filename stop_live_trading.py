
import os
import ccxt
import sys
from dotenv import load_dotenv

load_dotenv()

def stop_all():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("API credentials not found in .env")
        return

    ex = ccxt.binanceusdm({
        "apiKey": api_key,
        "secret": api_secret,
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })

    print("Checking Binance Futures...")
    
    # 1. Cancel all open orders
    try:
        open_orders = ex.fetch_open_orders()
        if open_orders:
            print(f"Cancelling {len(open_orders)} open orders...")
            for o in open_orders:
                ex.cancel_order(o['id'], o['symbol'])
        else:
            print("No open orders found.")
    except Exception as e:
        print(f"Error cancelling orders: {e}")

    # 2. Close all open positions
    try:
        balance = ex.fetch_balance()
        positions = [p for p in balance['info']['positions'] if abs(float(p['positionAmt'])) > 0]
        
        if positions:
            print(f"Closing {len(positions)} open positions...")
            for p in positions:
                symbol = p['symbol']
                qty = abs(float(p['positionAmt']))
                side = "sell" if float(p['positionAmt']) > 0 else "buy"
                print(f"Closing {symbol}: {side} {qty}")
                ex.create_market_order(symbol, side, qty, params={"reduceOnly": True})
        else:
            print("No open positions found.")
    except Exception as e:
        print(f"Error closing positions: {e}")

    # 3. Cancel algo orders (stop loss etc)
    try:
        algo_orders = ex.fapiPrivateGetOpenAlgoOrders()
        orders = algo_orders if isinstance(algo_orders, list) else algo_orders.get("orders", [])
        if orders:
            print(f"Cancelling {len(orders)} algo orders...")
            for o in orders:
                ex.fapiPrivateDeleteAlgoOrder({"algoId": int(o["algoId"])})
        else:
            print("No algo orders found.")
    except Exception as e:
        print(f"Error cancelling algo orders: {e}")

    print("Live trading stopped.")

if __name__ == "__main__":
    stop_all()
