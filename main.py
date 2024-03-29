import pandas as pd
import json
import altair as alt
import numpy as np
from altair import pipe, limit_rows, to_values
import streamlit as st
import time
import sys
from streamlit import cli as stcli

def main():
    st.set_page_config(page_title='Tuned JSON Viewer', page_icon="🔌", layout='wide', initial_sidebar_state='expanded')

    st.header("Upload a Tuned Backtest JSON file to populate charts!")

    alt.data_transformers.register('custom', lambda data: pipe(data, limit_rows(max_rows=10000), to_values))
    alt.data_transformers.enable('custom')
    alt.renderers.enable('altair_viewer')

    uploaded_file = st.file_uploader("Choose a file", type=['json'])
    # json_open = open('json.json')
    # uploaded_file = json.load(json_open)
    # # uploaded_file = open('json.json')
    option = st.selectbox(
        'What would you like to View? Batchtest, Backtest or Strategy Pack(Single Coin)',
        ('Backtest', 'Strategy Pack(Single Coin)'))

    if option == 'Backtest':
        if uploaded_file is not None:
            with st.spinner('Wait for it...'):
                time.sleep(3)
                st.balloons()
            st.success('Ready to Analyse!!')

            json_data = json.load(uploaded_file)

            domain = ['buy_hold', 'Cumulative_Profit']
            domain1 = ["Cumulative_Profit_Max", 'Cumulative_Profit_Min']
            domain2 = ['buy_hold', 'Strategy_Percentage']
            range_ = ['green', 'yellow']

            data1 = pd.json_normalize(json_data['trades'], record_path=['orders'], meta=['tradeNo'])
            print(data1.keys())
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
            merged['cumuProf'] = merged.profit.shift(fill_value=0).cumsum()
            merged['filledTime'] = pd.to_datetime(merged['filledTime'])
            merged['filledTimeM'] = pd.to_datetime(merged['filledTime'])
            merged['filledTimeD'] = pd.to_datetime(merged['filledTime'])
            merged['trade_duration'] = (merged['filledTime'] - merged.filledTime.shift(1)).dt.total_seconds() / 60 / 60
            start_alloc = merged["startAlloc"][0]
            first_filled = merged['filledPrice'][0]
            merged["cumBal"] = merged.apply(lambda x: x["startAlloc"] + x['cumuProf'], axis=1)
            cum_bal_coin = 0
            merged['cumBalCoin'] = merged.apply(lambda x: x['cumBal'] * x['filledPrice'], axis=1)
            result = merged.groupby([merged['filledTime'].dt.year, merged['filledTimeM'].dt.month])['cumBal'].last()
            start_price = merged['filledPrice'][0]
            merged['Winning_trade_counter'] = merged.apply(lambda x: 1.0
                                                           if float(x['profit']) > 0.0 else 0.0, axis=1)
            merged['Winning_trade_sum'] = merged.apply(lambda x: float(x['profit'])
                                                       if float(x['profit']) > 0.0 else 0.0, axis=1)

            merged['Winning_trade_counter_tot'] = merged['Winning_trade_counter'].cumsum()
            merged['Winning_trade_sum_tot'] = merged['Winning_trade_sum'].cumsum()
            merged['Average_winning_trade'] = merged.apply(lambda x:
                                                           x['Winning_trade_sum_tot'] / x['Winning_trade_counter_tot']
                                                           if x['Winning_trade_counter_tot'] > 0.0 else 0.0,
                                                           axis=1)

            merged['Losing_trade_counter'] = merged.apply(lambda x: 1.0
                                                          if float(x['profit']) < 0.0 else 0.0, axis=1)
            merged['Losing_trade_sum'] = merged.apply(lambda x: float(x['profit'])
                                                      if float(x['profit']) < 0.0 else 0.0, axis=1)

            merged['Losing_trade_counter_tot'] = merged['Losing_trade_counter'].cumsum()
            merged['Losing_trade_sum_tot'] = merged['Losing_trade_sum'].cumsum()
            merged['Average_losing_trade'] = merged.apply(lambda x:
                                                          x['Losing_trade_sum_tot'] / x['Losing_trade_counter_tot']
                                                          if x['Losing_trade_counter_tot'] > 0.0 else 0.0,
                                                          axis=1)

            merged['filled'] = merged['filledPrice'].ffill()
            merged["buy_hold"] = merged.apply(lambda x: x['filledPrice'] / start_price - 1.0, axis=1)
            merged['buy_hold_max'] = merged.buy_hold.shift(fill_value=0).cummax()
            merged["buy_hold_min"] = np.where((merged["buy_hold_max"] < merged["buy_hold_max"][1]),
                                              merged.loc[:, ["buy_hold"]].min(1),
                                              merged.loc[:, ["buy_hold"]].max(1),
                                              )
            merged['buy_hold_mdd'] = merged.apply(lambda x: 0.0 if x['buy_hold_max'] <=
                                                  0.0 or (1.0 - ((1.0 + x["buy_hold_min"]) /
                                                                 (1.0 + x['buy_hold_max']))) * -1.0 > 0.0
                                                  else (1.0 - ((1.0 + x["buy_hold_min"]) /
                                                               (1.0 + x['buy_hold_max']))) * -1.0,
                                                  axis=1)

            merged['profitableTrades'] = merged.apply(lambda x: 1.0 if float(x['profit']) > 0.0 else 0.0, axis=1)
            merged["Cumulative_Profit"] = merged.apply(lambda x: ((x['cumBal'] - x['startAlloc']) / x['startAlloc']), axis=1)
            merged['profitableTradesTot'] = merged['profitableTrades'].cumsum()
            print(start_price)

            merged['Profitable_Trades_Perc'] = merged.apply(lambda x: x['profitableTradesTot'] / (x['tradeNo']+1), axis=1)
            merged["Cumulative_Profit_Max"] = merged.Cumulative_Profit.shift(fill_value=0).cummax()
            merged["Cumulative_Profit_Min"] = np.where((merged["Cumulative_Profit_Max"] < merged["Cumulative_Profit_Max"][1]),
                                                       merged.loc[:, ["Cumulative_Profit"]].min(1),
                                                       merged.loc[:, ["Cumulative_Profit"]].max(1),
                                                       )

            merged["Cumulative_Profit_Max"] = merged.Cumulative_Profit.shift(fill_value=0).cummax()

            merged['worst_mdd'] = merged.apply(lambda x: 0.0 if x['Cumulative_Profit_Max'] <=
                                               0.0 else (1.0 - ((1.0 + x["Cumulative_Profit_Min"]) /
                                                                (1.0 + x['Cumulative_Profit_Max']))) * -1.0,
                                               axis=1)

            merged['worst_mdd'] = merged.worst_mdd.shift(fill_value=0).cummin()

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
            merged['Strategy_Percentage'] = merged.apply(
                lambda x: ((x['filledPrice'] * x['cumBal']) / (first_filled * startAlloc)) - 1.0,
                axis=1)
            nearest = alt.selection(type='single', nearest=True, on='mouseover',
                                    fields=['cumProf'], empty='none')

            nearest1 = alt.selection(type='single', nearest=True, on='mouseover',
                                    fields=['accumulatedBalance'], empty='none')

            result = result.reset_index()

            result['monthYear'] = "01" + "-" + result["filledTimeM"].astype(str) + "-" + result["filledTime"].astype(str)
            result['cumBalShift'] = result.cumBal.shift(1)
            result['cumBalShift'] = result['cumBalShift'].fillna(merged['startAlloc'])
            result['profit1'] = (result['cumBal'] / result['cumBalShift'] - 1.0)

            bars = alt.Chart(result, title=f'This chart shows you the monthly gains of {startAlloc[0]} '
                                        f'{data2["currencyPairDetails.settleCurrency"][1]}'
                            ).mark_bar().encode(
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
                                    offset=0)),
                color=alt.Color('key:N', scale={"range": ["yellow", "red"]})
            ).configure_view(
                strokeWidth=4,
                fill='#1c1c1e',
                stroke='#131313',
            ).properties().configure_axisY(
                labelAlign='right'
            )

            bars1 = alt.Chart(merged, title='This Chart shows the time between trade closes in Hours'
                            ).mark_bar().encode(
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
                                    offset=0)),
                color=alt.Color('key:N', scale={"range": ["yellow", "red"]})
            ).configure_view(
                strokeWidth=4,
                fill='#1c1c1e',
                stroke='#131313',
            ).properties().configure_axisY(
                labelAlign='right'
            )

            trades = alt.Chart(merged, title='This chart shows you the success rate over time'
                            ).transform_fold(
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
                color=alt.Color('key:N', scale={"range": ["yellow", "red"]})
                # alt.Color('key', scale={"range": ["yellow", "orange"]})  # color='key:N'
            ).configure_view(
                strokeWidth=4,
                fill='#1c1c1e',
                stroke='#131313',
            ).properties().configure_axisY(
                labelAlign='right'
            )

            chart = alt.Chart(merged,
                            title=f'This chart shows you the Accumulated % of {startAlloc[0]} '
                                    f'{data2["currencyPairDetails.settleCurrency"][1]}',
                            ).transform_fold(
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
                color=alt.Color('key:N', scale={"range": ["yellow", "red"]})
            ).configure_view(
                strokeWidth=4,
                fill='#1c1c1e',
                stroke='#131313',
            ).properties().configure_axisY(
                labelAlign='right'
            )

            chart3 = alt.Chart(merged, title=f'This Chart shows you the Max Drawdown from an equity High'
                               ).transform_fold(
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
                color=alt.Color('key:N', scale={"range": ["yellow", "red", "orange"]})
            ).configure_view(
                strokeWidth=4,
                fill='#1c1c1e',
                stroke='#131313',
            ).properties().configure_axisY(
                labelAlign='right'
            )

            chartmdd = alt.Chart(merged, title=f'This Chart shows you the Max Drawdown from an equity High versus '
                                               f'Buy and Hold Max Drawdown'
                               ).transform_fold(
                ['worst_mdd', 'buy_hold_mdd']).mark_line(
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
                color=alt.Color('key:N', scale={"range": ["yellow", "red", "orange"]})
            ).configure_view(
                strokeWidth=4,
                fill='#1c1c1e',
                stroke='#131313',
            ).properties().configure_axisY(
                labelAlign='right'
            )

            chart1 = alt.Chart(merged, title=f'This chart shows you the Accumulated Balance'
                                            f' of {startAlloc[0]} {data2["currencyPairDetails.settleCurrency"][1]}'
                            ).mark_line(
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
                                    labelOverlap=True)),
                color=alt.Color('key:N', scale={"range": ["yellow", "red"]})
            ).configure_view(
                strokeWidth=4,
                fill='#1c1c1e',
                stroke='#131313',
            ).properties().configure_axisY(
                labelAlign='right'
            )

            chart2 = alt.Chart(merged, title=f'This chart compares the Buy and Hold to the Strategy PnL '
                                            f'in {data2["currencyPairDetails.quote"][1]}'
                            ).transform_fold(
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
                color=alt.Color('key:N', scale={"range": ["yellow", "red"]})
            ).configure_view(
                strokeWidth=4,
                fill='#1c1c1e',
                stroke='#131313',
            ).properties().configure_axisY(
                labelAlign='right'
            )


            finalBal = merged['cumBal'].iloc[-1]
            finalPerc = merged['Cumulative_Profit'].iloc[-1] * 100.0
            finalBnH = merged['buy_hold'].iloc[-1] * 100.0
            finalMDD = (merged['worst_mdd'].iloc[-1] * 100.0)
            buyholdMDD = (merged['buy_hold_mdd'].iloc[-1] * 100.0)

            avg_win_trade = (merged['Average_winning_trade'].iloc[-1]) * 100.0
            avg_lose_trade = (merged['Average_losing_trade'].iloc[-1]) * 100.0

            st.subheader(f'{titleData["name"][0]} - {titleData["type"][0]} - Trading {coinData["coinPair"][0]} '
                        f'on {coinData["exchange"][0]}')
            st.altair_chart(chart, use_container_width=True)
            st.write(f'Profit and Loss Percentage at end of Backtest = {round(finalPerc, 2)}%')
            st.write(f'Buy and Hold Percentage at end of Backtest = {round(finalBnH, 2)}%')
            check = (coinData['exchange'][0] == 'BYBIT' or
                    coinData['exchange'][0] == 'BINANCE_COIN_FUTURES' or
                    coinData['exchange'][0] == 'HUOBI_COIN_SWAPS' or
                    coinData['exchange'][0] == 'BITMEX')

            if check:
                expander = st.expander(f'If using Coin Futures - Click here to see the '
                                    f'{data2["currencyPairDetails.quote"][1]} Comparisons')
                expander.altair_chart(chart2, use_container_width=True)
                expander.write(f"{startAlloc}")
                expander.write(f"{first_filled}")

            st.altair_chart(chart3, use_container_width=True)
            st.altair_chart(chartmdd, use_container_width=True)
            st.write(f'The worst Realised Max Drawdown you incurred was {round(finalMDD, 2)}%')
            st.write(f'Buy and Hold Max Drawdown was {round(buyholdMDD, 2)}%')
            st.altair_chart(chart1, use_container_width=True)
            st.altair_chart(bars, use_container_width=True)
            st.altair_chart(bars1, use_container_width=True)

            try:
                number = st.number_input('Length of the Moving Average to be used on the Profitable Trades Percentage',
                                         value=50)
                if number is None or number <= 0:
                    st.text(f'Pick a length greater than 0 for the moving average to populate chart')
                else:
                    merged['Profitable_Trades_Avg'] = merged['profitableTrades'].rolling(window=number, min_periods=1).mean()
                    st.altair_chart(trades, use_container_width=True)
            except:
                st.text(f'Please select a Moving Average Length...')

            st.text(f'Initial Allocation: {startAlloc[0]} {data2["currencyPairDetails.settleCurrency"][1]}')
            st.text(f'Final Balance: {round(finalBal, 2)} {data2["currencyPairDetails.settleCurrency"][1]}')
            st.text(f'Total Commission Paid: {round(commSum, 2)} in {data2["currencyPairDetails.settleCurrency"][1]}')
            st.text(f'Total Number of Trade: {merged["tradeNo"].iloc[-1]}')
            st.text(f'Average winning trade Percentage: {round(avg_win_trade, 2)}%')
            st.text(f'Average losing trade Percentage: {round(avg_lose_trade, 2)}%')
            # merged.to_csv('out.csv')
        else:
            st.text("JSON not uploaded")

    if option == 'Batchtest':
        if uploaded_file is not None:
            with st.spinner('Wait for it...'):
                time.sleep(3)
                st.balloons()
            st.success('Ready to Analyse!!')

            json_data_batch = json.load(uploaded_file)
            batch_data = pd.json_normalize(data=json_data_batch)
            batch_data_name = pd.json_normalize(
                json_data_batch, 'strategyParameters', ['performance'])
            print(batch_data_name)
            # print(batch_data.keys())
            # print(batch_data)
            #
            # batch_data.to_csv('name.csv')

    if option == 'Strategy Pack(Single Coin)':
        st.text(f'Work in Progress')


if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
