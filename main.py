import pandas as pd
from altair.examples.ranged_dot_plot import chart
import json
import altair as alt
import numpy as np
import altair_viewer
from altair import pipe, limit_rows, to_values
import streamlit as st
from vega_datasets import data

st.set_page_config(page_title='Dashboard', page_icon="🔌", layout='wide', initial_sidebar_state='expanded')

st.header("Upload a Tuned Backtest JSON file to populate charts!")

alt.data_transformers.register('custom', lambda data: pipe(data, limit_rows(max_rows=10000), to_values))
alt.data_transformers.enable('custom')
alt.renderers.enable('altair_viewer')


uploaded_file = st.file_uploader("Choose a file", type=['json'])

# uploaded_file = open('json.json')

if uploaded_file is not None:
    st.balloons()
    json_data = json.load(uploaded_file)

    domain = ['buy_hold', 'Cumulative_Profit']
    domain1 = ["Cumulative_Profit_Max", 'Cumulative_Profit_Min']
    domain2 = ['buy_hold', 'Strategy_Percentage']
    range_ = ['green', 'yellow']

    data1 = pd.json_normalize(json_data['trades'], record_path=['orders'], meta=['tradeNo'])
    data2 = pd.json_normalize(json_data, record_path=['trades'])
    titleData = pd.json_normalize(json_data['strategy'])
    coinData = pd.json_normalize(json_data['market'])
    data3 = pd.json_normalize(json_data)

    data3.drop(data3.columns.difference(['performance.startAllocation']), 1, inplace=True)

    str_filter = ['Sell', 'CloseLong', 'CloseShort']
    buy_filter = ['Buy', 'Long', 'Short']

    filtered = data1[data1['side'].isin(str_filter)]

    data1['adjComm'] = np.where(data1['side'] == 'Buy',
                                data1['commissionPaid'] * data1['filledPrice'],
                                data1['commissionPaid'])

    commSum = data1['adjComm'].sum(axis=0)

    data2.drop(data2.columns.difference(['tradeNo', 'profit', 'profitPercentage', 'accumulatedBalance',
                                         'currencyPairDetails.quote', 'currencyPairDetails.base', 'compoundProfitPerc',
                                         'strategyCompoundProfitPerc', 'currencyPairDetails.settleCurrency']),
               1, inplace=True)

    merged = pd.merge(filtered, data2)
    merged.drop(merged.columns.difference(['side', 'tradeNo', 'filledTime', 'filledPrice',
                                           'profit', 'profitPercentage',
                                           'accumulatedBalance',
                                           'compoundProfitPerc', 'strategyCompoundProfitPerc',
                                           'currencyPairDetails.base', 'currencyPairDetails.settleCurrency',
                                           'currencyPairDetails.quote', 'startAlloc', 'cumBal', 'cumProf']),
                1, inplace=True)

    merged["startAlloc"] = pd.Series([data3['performance.startAllocation'][0] for x in range(len(merged.index))])

    diff = merged["startAlloc"][0]

    merged['filledTime'] = pd.to_datetime(merged['filledTime'])
    merged['filledTimeM'] = pd.to_datetime(merged['filledTime'])
    merged['filledTimeD'] = pd.to_datetime(merged['filledTime'])
    # merged['filledTimeD'] = merged['filledTimeD'].dt.month

    merged['trade_duration'] = (merged['filledTime'] - merged.filledTime.shift(1)).dt.total_seconds() / 60 / 60

    def get_cum_bal(start_alloc, profit):
        global diff
        if diff == start_alloc:
            diff = start_alloc + profit
        else:
            diff += profit

        return diff


    merged["cumBal"] = merged.apply(lambda x: get_cum_bal(x['startAlloc'], x['profit']), axis=1)
    cum_bal_coin = 0

    def get_coin_bal(cum_bal, filled_price):
        global cum_bal_coin
        cum_bal_coin = cum_bal * filled_price
        return cum_bal_coin


    merged['cumBalCoin'] = merged.apply(lambda x: get_coin_bal(x['cumBal'], x['filledPrice']), axis=1)

    result = merged.groupby([merged['filledTime'].dt.year, merged['filledTimeM'].dt.month])['cumBal'].last()

    start_price = merged['filledPrice'][0]

    def get_buy_hold(current_price):
        global start_price
        buy_hold = current_price / start_price - 1
        return buy_hold

    merged["buy_hold"] = merged.apply(lambda x: get_buy_hold(x['filledPrice']), axis=1)

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

    merged['profitableTrades'] = merged.apply(lambda x: get_prof_trades(x['profit']), axis=1)

    merged["Cumulative_Profit"] = merged.apply(lambda x: get_profit(x['cumBal'], x['startAlloc']), axis=1)

    merged['profitableTradesTot'] = merged['profitableTrades'].cumsum()

    def get_coin_perc(buy_hold, profit):
        coin_perc = buy_hold * (1.0 + profit)
        return coin_perc

    merged['Strategy_Percentage'] = merged.apply(lambda x: get_coin_perc(x["buy_hold"], x['Cumulative_Profit']), axis=1)

    def get_prof_trades_tot(total_trades, winning_trades):
        trades_perc = int(winning_trades) / int(total_trades)  # * 100.0
        return trades_perc

    merged['Profitable_Trades_Perc'] = merged.apply(lambda x: get_prof_trades_tot(x['tradeNo']+1,
                                                                                  x['profitableTradesTot']),
                                                    axis=1)

    merged["Cumulative_Profit_Max"] = merged.Cumulative_Profit.shift(fill_value=0).cummax()

    merged["Cumulative_Profit_Min"] = np.where((merged["Cumulative_Profit_Max"] < merged["Cumulative_Profit_Max"][1]),
                                               merged.loc[:, ["Cumulative_Profit"]].min(1),
                                               merged.loc[:, ["Cumulative_Profit"]].max(1),
                                               )

    merged["Cumulative_Profit_Max"] = merged.Cumulative_Profit.shift(fill_value=0).cummax()

    coin = 'na'
    for col in data2:
        if col == 'currencyPairDetails.quote':
            coin = data2[col]
        else:
            coin = 'na'

    coin1 = 'na'
    for col2 in data2:
        if col2 == 'currencyPairDetails.base':
            coin1 = data2[col2]
        else:
            coin1 = 'na'

    startAlloc = 'na'

    for col1 in data3:
        if col1 == 'performance.startAllocation':
            startAlloc = data3[col1]
        else:
            startAlloc = 'na'

    merged['maxValPerc'] = merged['Cumulative_Profit'].max()
    merged['minValPerc'] = merged['Cumulative_Profit'].min()

    y_range_max_1 = merged['maxValPerc'].max()
    y_range_min_1 = merged['minValPerc'].min()

    x_range_max_1 = merged['filledTime'].max()
    x_range_min_1 = merged['filledTime'].min()

    merged['maxValBal'] = merged["cumBal"].max()
    merged['minValBal'] = merged["cumBal"].min()

    y_range_max_2 = merged['maxValBal'].max()
    y_range_min_2 = merged['minValBal'].min()

    x_range_max_2 = merged['filledTime'].max()
    x_range_min_2 = merged['filledTime'].min()

    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['cumProf'], empty='none')

    nearest1 = alt.selection(type='single', nearest=True, on='mouseover',
                             fields=['accumulatedBalance'], empty='none')

    result = result.reset_index()

    result['monthYear'] = "01" + "-" + result["filledTimeM"].astype(str) + "-" + result["filledTime"].astype(str)

    result['cumBalShift'] = result.cumBal.shift(1)
    result['cumBalShift'] = result['cumBalShift'].fillna(merged['startAlloc'])

    result['profit1'] = (result['cumBal'] / result['cumBalShift'] - 1.0)

    bars = alt.Chart(result).mark_bar().encode(
        x=alt.X('monthYear:O', sort=alt.EncodingSortField(field="monthYear", op='count', order='ascending'),
                scale=alt.Scale(nice=False),
                axis=alt.Axis(formatType="timeUnit", title='Date',
                              labelAngle=-70,
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)
                ),
        y=alt.Y('profit1', scale=alt.Scale(nice=False),
                axis=alt.Axis(title=f'Monthly Percentage', grid=True, format='%',
                              offset=0))
    )

    bars1 = alt.Chart(merged).mark_bar().encode(
        x=alt.X('tradeNo:O', sort=alt.EncodingSortField(field="monthYear", op='count', order='ascending'),
                scale=alt.Scale(nice=False),
                axis=alt.Axis(formatType="timeUnit", title='Date',
                              labelAngle=-70,
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)
                ),
        y=alt.Y('trade_duration', scale=alt.Scale(nice=False),
                axis=alt.Axis(title=f'Average Time in Hours between Trade Close', grid=True,
                              offset=0))
    )

    trades = alt.Chart(merged).transform_fold(
        ['Profitable_Trades_Perc', 'Profitable_Trades_Avg']).mark_line(
        interpolate='basis',
        line={'color': 'yellow'},
        opacity=0.5
        ).encode(
        x=alt.X('filledTime:T', scale=alt.Scale(nice=False),
                axis=alt.Axis(formatType="timeUnit", format="%B of %Y", title='Date',
                              labelAngle=-70,
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)),
        y=alt.Y('value:Q',
                axis=alt.Axis(title=f'Profitable Trades Percentage', format='%',
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)),
        color=alt.Color('key:N', scale={"range":["yellow", "red"]})  # alt.Color('key', scale={"range": ["yellow", "orange"]})  # color='key:N'
    )

    chart.properties().configure_axisY(
            titleAngle=0,
            titleY=-10,
            titleX=-60,
            labelPadding=160,
            labelAlign='left'
        )

    chart = alt.Chart(merged).transform_fold(
        ['buy_hold', 'Cumulative_Profit']).mark_line(
        interpolate='basis',
        line={'color': 'yellow'},
        opacity=0.5
    ).encode(
        x=alt.X('filledTime:T', scale=alt.Scale(nice=False),
                axis=alt.Axis(formatType="timeUnit", format="%B of %Y", title='Date',
                              labelAngle=-70,
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)),
        y=alt.Y('value:Q', scale=alt.Scale(nice=False),
                axis=alt.Axis(labelSeparation=3, format='%',
                              labelPadding=0,
                              labelOverlap=True)),
        color=alt.Color('key:N', scale={"range":["yellow", "red"]})
    )

    chart3 = alt.Chart(merged).transform_fold(
        ["Cumulative_Profit_Max", 'Cumulative_Profit_Min']).mark_line(
        interpolate='basis',
        line={'color': 'yellow'},
        opacity=0.5
    ).encode(
        x=alt.X('filledTime:T', scale=alt.Scale(nice=False),
                axis=alt.Axis(formatType="timeUnit", format="%B of %Y", title='Date',
                              labelAngle=-70,
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)),
        y=alt.Y('value:Q', scale=alt.Scale(nice=False),
                axis=alt.Axis(labelSeparation=3, format='%',
                              labelPadding=0,
                              labelOverlap=True)),
        color=alt.Color('key:N', scale={"range":["yellow", "red"]})
    )

    chart1 = alt.Chart(merged).mark_line(
        interpolate='basis',
        line={'color': 'yellow'},
        opacity=0.5
        ).encode(
        x=alt.X('filledTime:T', scale=alt.Scale(nice=False),
                axis=alt.Axis(formatType="timeUnit", format="%B of %Y", title='Date',
                              labelAngle=-70,
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)),
        y=alt.Y('cumBal', scale=alt.Scale(nice=False),
                axis=alt.Axis(title=f'Accumulated Balance of {startAlloc[0]} '
                                    f'{data2["currencyPairDetails.settleCurrency"][1]}',
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True))
    )

    chart2 = alt.Chart(merged).transform_fold(
        ['buy_hold', 'Strategy_Percentage']).mark_line(
        interpolate='basis',
        line={'color': 'yellow'},
        opacity=0.5
        ).encode(
        x=alt.X('filledTime:T', scale=alt.Scale(nice=False),
                axis=alt.Axis(formatType="timeUnit", format="%B of %Y", title='Date',
                              labelAngle=-70,
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)),
        y=alt.Y('value:Q', scale=alt.Scale(nice=False),
                axis=alt.Axis(title=f'Percentage comparison versus Buy and hold', format='%',
                              labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)),
        color=alt.Color('key:N', scale={"range":["yellow", "red"]})
    )
    #
    # selectors = alt.Chart(merged).mark_point().encode(
    #     x='filledTime:T',
    #     opacity=alt.value(0),
    # ).add_selection(
    #     nearest
    # )
    #
    # # Draw points on the line, and highlight based on selection
    # points = chart.mark_point().encode(
    #     opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    # )
    #
    # # Draw text labels near the points, and highlight based on selection
    # text = chart.mark_text(align='left', dx=5, dy=-5).encode(
    #     text=alt.condition(nearest, 'cumProf', alt.value(' '))
    # )
    #
    # # Draw a rule at the location of the selection
    # rules = alt.Chart(merged).mark_rule(color='gray').encode(
    #     x='filledTime:T',
    # ).transform_filter(
    #     nearest
    # )
    #
    # # Put the five layers into a chart and bind the data
    # plot = alt.layer(
    #     chart.mark_line(color='blue').encode(
    #         y=alt.Y('cumProf', scale=alt.Scale(nice=False),
    #                 axis=alt.Axis(title=f'Accumulated % of {startAlloc[0]} '
    #                                     f'{data2["currencyPairDetails.settleCurrency"][1]}',
    #                               grid=True, format='%',
    #                               offset=0))),
    #     selectors,
    #     points,
    #     text,
    #     rules
    # )

    finalBal = merged['cumBal'].iloc[-1]

    st.subheader(f'{titleData["name"][0]} - {titleData["type"][0]} - Trading {coinData["coinPair"][0]} '
                 f'on {coinData["exchange"][0]}')
    st.subheader(f'This chart shows you the Accumulated % of {startAlloc[0]} '
                 f'{data2["currencyPairDetails.settleCurrency"][1]}')
    st.altair_chart(chart, use_container_width=True)
    st.subheader(f'This Chart shows you the Max Drawdown from an equity High')
    st.altair_chart(chart3, use_container_width=True)

    check = (coinData['exchange'][0] == 'BYBIT' or
             coinData['exchange'][0] == 'BINANCE_COIN_FUTURES' or
             coinData['exchange'][0] == 'HUOBI_COIN_SWAPS' or
             coinData['exchange'][0] == 'BITMEX')

    if check:
        expander = st.expander(f'If using Coin Futures - Click here to see the '
                               f'{data2["currencyPairDetails.quote"][1]} Comparisons')
        expander.subheader(f'This chart compares the Buy and Hold to the Strategy PnL '
                           f'in {data2["currencyPairDetails.quote"][1]}')
        expander.altair_chart(chart2, use_container_width=True)

    st.subheader(f'This chart shows you the Accumulated Balance'
                 f' of {startAlloc[0]} {data2["currencyPairDetails.settleCurrency"][1]}')
    st.altair_chart(chart1, use_container_width=True)
    st.subheader(f'This chart shows you the monthly gains of {startAlloc[0]} '
                 f'{data2["currencyPairDetails.settleCurrency"][1]}')
    st.altair_chart(bars, use_container_width=True)

    st.subheader(f'This Chart shows the time between trade closes in Hours')
    st.altair_chart(bars1, use_container_width=True)

    st.subheader('This chart shows you the success rate over time')

    number = st.number_input('Length of the Moving Average to be used on the Profitable Trades Percentage', value=50)

    merged['Profitable_Trades_Avg'] = merged['profitableTrades'].rolling(window=number, min_periods=1).mean()

    st.altair_chart(trades, use_container_width=True)
    st.text(f'Initial Allocation: {startAlloc[0]} {data2["currencyPairDetails.settleCurrency"][1]}')
    st.text(f'Final Balance: {round(finalBal, 2)} {data2["currencyPairDetails.settleCurrency"][1]}')
    st.text(f'Total Commission Paid: {round(commSum, 2)} in {data2["currencyPairDetails.settleCurrency"][1]}')
    st.text(f'Total Number of Trade: {merged["tradeNo"].iloc[-1]}')

else:
    st.text("JSON not uploaded")
