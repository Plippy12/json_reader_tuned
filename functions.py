import altair as alt
from altair import pipe, limit_rows, to_values
import altair_viewer
# All functions to be placed within this file


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

