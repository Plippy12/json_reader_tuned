import pandas as pd
from altair.examples.ranged_dot_plot import chart
import json
import altair as alt
import altair_viewer
from altair import pipe, limit_rows, to_values
import streamlit as st

st.set_page_config(page_title='Dashboard', page_icon="🔌", layout='wide', initial_sidebar_state='expanded')

st.header("Upload a JSON file to populate charts!")

alt.data_transformers.register('custom', lambda data: pipe(data, limit_rows(max_rows=10000), to_values))
alt.data_transformers.enable('custom')
alt.renderers.enable('altair_viewer')


uploaded_file = st.file_uploader("Choose a file", type=['json'])


if uploaded_file is not None:

    json_data = json.load(uploaded_file)

    data1 = pd.json_normalize(json_data['trades'], record_path=['orders'], meta=['tradeNo'])
    data2 = pd.json_normalize(json_data, record_path=['trades'])
    titleData = pd.json_normalize(json_data['strategy'])
    coinData = pd.json_normalize(json_data['market'])
    data3 = pd.json_normalize(json_data)

    data3.drop(data3.columns.difference(['performance.startAllocation']), 1, inplace=True)

    isSell = ['Sell']
    filtered = data1[data1['side'].isin(isSell)]

    data2.drop(data2.columns.difference(['tradeNo', 'profit', 'profitPercentage', 'accumulatedBalance',
                                         'currencyPairDetails.quote', 'compoundProfitPerc',
                                         'strategyCompoundProfitPerc']), 1, inplace=True)


    merged = pd.merge(filtered, data2)
    merged.drop(merged.columns.difference(['side', 'filledTime', 'profit', 'profitPercentage', 'accumulatedBalance',
                                           'compoundProfitPerc', 'strategyCompoundProfitPerc',
                                           'currencyPairDetails.quote', 'startAlloc', 'cumBal', 'cumProf']),
                1, inplace=True)

    merged["startAlloc"] = pd.Series([data3['performance.startAllocation'][0] for x in range(len(merged.index))])

    diff = merged["startAlloc"][0]

    merged['filledTime'] = pd.to_datetime(merged['filledTime'])
    merged['filledTimeM'] = pd.to_datetime(merged['filledTime'])
    merged['filledTimeD'] = pd.to_datetime(merged['filledTime'])

    result = merged.groupby([merged['filledTime'].dt.year, merged['filledTimeM'].dt.month]).agg({'profit':sum})


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


    merged["cumProf"] = merged.apply(lambda x: get_profit(x['cumBal'], x['startAlloc']), axis=1)


    # print("data1 is:", data1.keys())
    # print("data2 is:", data2.keys())
    # print("data3 is:", data3.keys())
    # print("filtered is:", filtered.keys())
    print("merged is:", merged.keys())



    coin = 'na'
    for col in data2:
        if col == 'currencyPairDetails.quote':
            coin = data2[col]
        else:
            coin = 'na'

    startAlloc = 'na'

    for col1 in data3:
        if col1 == 'performance.startAllocation':
            startAlloc = data3[col1]
        else:
            startAlloc = 'na'


    chart.properties(width=700).configure_axisY(
        titleAngle=0,
        titleY=-10,
        titleX=-60,
        labelPadding=160,
        labelAlign='left'
    )


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
                axis=alt.Axis(title=f'Accumulated % of {startAlloc[0]} {coin[1]} Per Month', grid=True, #, format='%.2f'
                              offset=0))
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
        title=f'{titleData["name"][0]} - {titleData["type"][0]} - Trading {coinData["coinPair"][0]}'
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
                    axis=alt.Axis(title=f'Accumulated % of {startAlloc[0]} {coin[1]}', grid=True, format='%',
                                  offset=0))),
        selectors,
        points,
        text,
        rules
    ).properties(
        width=600,
        height=400
    )

    plot2 = alt.vconcat(plot, bars)

    st.altair_chart(plot2, use_container_width=True)

else:
    st.text("JSON not uploaded")
