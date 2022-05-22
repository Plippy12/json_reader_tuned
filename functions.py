

# All functions to be placed within this file

def get_cum_bal(start_alloc, profit):
    start_alloc1 = start_alloc + profit
    return start_alloc1


def get_coin_bal(cum_bal_coin, cum_bal, filled_price):
    cum_bal_coin = cum_bal * filled_price
    return cum_bal_coin


def get_buy_hold(start_price, current_price):
    buy_hold = current_price / start_price - 1
    return buy_hold


def get_profit(cum_bal, start_alloc):
    inc = cum_bal - start_alloc
    prof = inc / start_alloc
    return prof


def get_prof_trades(profit):
    if float(profit) > 0:
        profit_check = 1
    else:
        profit_check = 0
    return profit_check


def get_coin_perc(buy_hold, profit):
    coin_perc = buy_hold * (1.0 + profit)
    return coin_perc


def get_prof_trades_tot(total_trades, winning_trades):
    trades_perc = int(winning_trades) / int(total_trades)  # * 100.0
    return trades_perc


