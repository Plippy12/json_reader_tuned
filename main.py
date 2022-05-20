import pandas as pd
from altair.examples.ranged_dot_plot import chart
import json
import altair as alt
import numpy as np
import altair_viewer
from altair import pipe, limit_rows, to_values
import streamlit as st

st.set_page_config(page_title='Dashboard', page_icon="🔌", layout='wide', initial_sidebar_state='expanded')

st.header("Upload a Tuned Backtest JSON file to populate charts!")

alt.data_transformers.register('custom', lambda data: pipe(data, limit_rows(max_rows=10000), to_values))
alt.data_transformers.enable('custom')
alt.renderers.enable('altair_viewer')


uploaded_file = st.file_uploader("Choose a file", type=['json'])

# uploaded_file = open('json.json')

if uploaded_file is not None:

    json_data = json.load(uploaded_file)

    data1 = pd.json_normalize(json_data['trades'], record_path=['orders'], meta=['tradeNo'])
    data2 = pd.json_normalize(json_data, record_path=['trades'])
    titleData = pd.json_normalize(json_data['strategy'])
    coinData = pd.json_normalize(json_data['market'])
    data3 = pd.json_normalize(json_data)

    data3.drop(data3.columns.difference(['performance.startAllocation']), 1, inplace=True)

    isSell = ['Sell']
    closeLong = ['CloseLong']
    closeShort = ['CloseShort']
    strFilt = ['Sell', 'CloseLong', 'CloseShort']
    buyFilt = ['Buy']
    filtered = data1[data1['side'].isin(strFilt)]
    filteredCommBuy = data1[data1['side'].isin(buyFilt)]
    data1['adjComm'] = adjustedCommBuy = np.where(data1['side'] == 'Buy',
                                                  data1['commissionPaid'] * data1['filledPrice'],
                                                  data1['commissionPaid'])

    print(data1['adjComm'])

    commSum = data1['adjComm'].sum(axis=0)

    data2.drop(data2.columns.difference(['tradeNo', 'profit', 'profitPercentage', 'accumulatedBalance',
                                         'currencyPairDetails.quote', 'currencyPairDetails.base', 'compoundProfitPerc',
                                         'strategyCompoundProfitPerc', 'currencyPairDetails.settleCurrency']),
               1, inplace=True)

    merged = pd.merge(filtered, data2)
    merged.drop(merged.columns.difference(['side', 'tradeNo', 'filledTime', 'profit', 'profitPercentage',
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

    result = merged.groupby([merged['filledTime'].dt.year, merged['filledTimeM'].dt.month]).agg({'profit': sum})

    def get_cumBal(startAlloc, profit):
        global diff
        if diff == startAlloc:
            diff = startAlloc + profit
        else:
            diff += profit

        return diff


    merged["cumBal"] = merged.apply(lambda x: get_cumBal(x['startAlloc'], x['profit']), axis=1)


    def get_profit(cumBal, startAlloc):
        inc = cumBal - startAlloc
        prof = inc / startAlloc
        return prof

    def get_proftrades(profit):
        if float(profit) > 0:
            profit_check = 1
        else:
            profit_check = 0
        return profit_check


    merged['profitableTrades'] = merged.apply(lambda x: get_proftrades(x['profit']), axis=1)

    merged["cumProf"] = merged.apply(lambda x: get_profit(x['cumBal'], x['startAlloc']), axis=1)

    merged['profitableTradesTot'] = merged['profitableTrades'].cumsum()


    def get_proftradesTot(totalTrades, winningTrades):
        tradesPerc = int(winningTrades) / int(totalTrades) * 100.0
        return tradesPerc

    merged['profitableTradesRolSum'] = merged.apply(lambda x: get_proftradesTot(x['tradeNo']+1,
                                                                                x['profitableTradesTot']),
                                                    axis=1)

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



    merged['maxValPerc'] = merged['cumProf'].max()
    merged['minValPerc'] = merged['cumProf'].min()

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

    result.reset_index(inplace=True)
    result['monthYear'] = "01" + "-" + result["filledTimeM"].astype(str) + "-" + result["filledTime"].astype(str)
    result['profit1'] = result['profit'] * 100.0

    # merged.to_csv('Trade-Data.csv')
    # result.to_csv('Monthly-Data.csv')

    bars = alt.Chart(result).mark_bar().encode(
        x=alt.X('monthYear:O', sort=alt.EncodingSortField(field="monthYear", op='count', order='ascending'),
                scale=alt.Scale(nice=False),
                axis=alt.Axis(formatType="timeUnit", title='Date')),
        y=alt.Y('profit1', scale=alt.Scale(nice=False),
                axis=alt.Axis(title=f'Accumulated % of {startAlloc[0]} '
                                    f'{data2["currencyPairDetails.quote"][1]} Per Month', grid=True,
                              offset=0))
    ).properties(
        width=1000,
        height=600
    )

    trades = alt.Chart(merged).mark_line(
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
        y=alt.Y('profitableTradesRolSum',
                axis=alt.Axis(title=f'Profitable Trades Percentage', labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)),
    ).properties(
        width=1000,
        height=600
    )


    chart.properties().configure_axisY(
            titleAngle=0,
            titleY=-10,
            titleX=-60,
            labelPadding=160,
            labelAlign='left'
        )

    chart = alt.Chart(merged).mark_line(
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
        y=alt.Y('cumProf', scale=alt.Scale(nice=False),
                axis=alt.Axis(labelSeparation=3,
                              labelPadding=0,
                              labelOverlap=True)),
    ).properties(
        title=f'{titleData["name"][0]} - {titleData["type"][0]} - Trading {coinData["coinPair"][0]}',
        width = 1000,
        height = 600
    )

    selectors = alt.Chart(merged).mark_point().encode(
        x='filledTime:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = chart.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = chart.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'cumProf', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(merged).mark_rule(color='gray').encode(
        x='filledTime:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    plot = alt.layer(
        chart.mark_line(color='blue').encode(
            y=alt.Y('cumProf', scale=alt.Scale(nice=False),
                    axis=alt.Axis(title=f'Accumulated % of {startAlloc[0]} {data2["currencyPairDetails.quote"][1]}',
                                  grid=True, format='%',
                                  offset=0))),
        selectors,
        points,
        text,
        rules
    ).properties(
        width=600,
        height=400
    )

    plot2 = alt.vconcat(plot, bars, trades)

    st.subheader(f'This chart shows you the Accumulated % of {startAlloc[0]} {data2["currencyPairDetails.quote"][1]}')
    st.altair_chart(plot, use_container_width=True)
    st.subheader(f'This chart shows you the monthly gains of {startAlloc[0]} {data2["currencyPairDetails.quote"][1]}')
    st.altair_chart(bars, use_container_width=True)
    st.subheader('This chart shows you the success rate over time')
    st.altair_chart(trades, use_container_width=True)
    st.text(f'Total Commission Paid: {round(commSum, 2)} in {data2["currencyPairDetails.settleCurrency"][1]}')

else:
    st.text("JSON not uploaded")
