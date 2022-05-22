import altair as alt

# All functions to be placed within this file

def get_cum_bal(start_alloc, profit):
    start_alloc += profit
    return start_alloc


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


def gen_chart(data, title, line1, line2, x_axis, y_axis):
    alt.Chart(data,
              title=title,
              ).transform_fold(
        [line1, line2]).mark_line(
        interpolate='basis',
        opacity=0.5
    ).encode(
        x=alt.X(x_axis, scale=alt.Scale(nice=False),
                axis=alt.Axis(formatType="timeUnit", format="%B of %Y", title='Date',
                              labelAngle=-70,
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)),
        y=alt.Y(y_axis, scale=alt.Scale(nice=False),
                axis=alt.Axis(labelSeparation=3, format='%',
                              labelPadding=0,
                              labelOverlap=True)),
        color=alt.Color('key:N', scale={"range": ["yellow", "red"]})
    ).configure_view(
        strokeWidth=4,
        fill='#1c1c1e',
        stroke='#131313',
    ).properties().configure_axisY(
        labelAlign='right'
    )

