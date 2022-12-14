import numpy as np
import pandas as pd
import geopandas
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime, time

# settings
pd.set_option('display.float_format', lambda x: '%.2f' % x)
st.set_page_config(layout="wide")


# get geofile
@st.cache( allow_output_mutation=True )
def get_geofile( url ):
    geofile = geopandas.read_file(url)

    return geofile

# read data
@st.cache( allow_output_mutation=True )
def get_data( path ):
    try:
        data = pd.read_csv( path )

        #adjustments
        data['date'] = pd.to_datetime( data['date'] )
        #data['year'] = pd.to_datetime(data['date']).dt.strftime('%Y')
        data.drop(data[data.bedrooms == 33].index, inplace=True) # typo
        data['bathrooms'] = np.round(data['bathrooms']).astype(int)
        data['floors'] = np.round(data['floors']).astype(int)

        #new columns
        data.round({'price': 2})
        data['price_sqft'] = data['price'] / data['sqft_lot']

        full_data = data.copy()
        #consider only most recent entry for same ids
        new_data = data.sort_values('date').drop_duplicates('id',keep='last')

        return new_data, full_data
    except:
        st.write( 'Error in obtaining data.' )
        return None

def define_filters(data):
    st.sidebar.title("Filter Options")

    # column selection (mandatory)
    # variables = st.sidebar.multiselect(
    #     'Columns', 
    #     data.columns.unique() 
    # )
    variables = data.columns.unique() 

    # zipcode filter list
    zipcodes = st.sidebar.multiselect(
        'Zipcode', 
        data['zipcode'].sort_values().unique()
    )
    if len(zipcodes) == 0:
        zipcodes = data['zipcode'].sort_values().unique() # all zipcodes

    bedrooms = st.sidebar.multiselect(
        'Bedrooms', 
        data.bedrooms.sort_values().unique() 
    )
    if len(bedrooms) == 0:
        bedrooms = data['bedrooms'].sort_values().unique()

    bathrooms = st.sidebar.multiselect(
        'Bathrooms', 
        data.bathrooms.unique() 
    )
    if len(bathrooms) == 0:
        bathrooms = data['bathrooms'].sort_values().unique()

    floors = st.sidebar.multiselect(
        'Floors', 
        data.floors.unique() 
    )
    if len(floors) == 0:
        floors = data['floors'].sort_values().unique()

    condition = st.sidebar.multiselect(
        'Condition', 
        data.condition.sort_values().unique() 
    )
    if len(condition) == 0:
        condition = data['condition'].sort_values().unique()

    waterfront = st.sidebar.selectbox(
        'Waterfront?',
        ('All apply', 'Yes', 'No')
    )
    if waterfront == 'Yes':
        waterfront = [1]
    elif waterfront == 'No':
        waterfront = [0]
    else:
        waterfront = data['waterfront'].sort_values().unique()

    renovated = st.sidebar.selectbox(
        'Renovated?',
        ('All apply', 'Yes', 'No')
    )
    if renovated == 'Yes':
        renovated = np.delete(data['yr_renovated'].sort_values().unique(), np.where(data['yr_renovated'].unique() == 0))
    elif renovated == 'No':
        renovated = [0]
    else:
        renovated = data['yr_renovated'].sort_values().unique()

    return (variables, zipcodes, bedrooms, bathrooms, floors, waterfront, condition, renovated)

def set_filters(data, zipcodes, bedrooms, bathrooms, floors, waterfront, condition, renovated):

    df = data[
        (data['zipcode'].isin(zipcodes))
        & (data['bedrooms'].isin(bedrooms))
        & (data['bathrooms'].isin(bathrooms))
        & (data['floors'].isin(floors))
        & (data['waterfront'].isin(waterfront))
        & (data['condition'].isin(condition))
        & (data['yr_renovated'].isin(renovated))
    ].reset_index()

    return df

def display_basic_info(df, variables):
    st.write( 'Displaying first 15 rows:' )

    # show filtered/simplified dataframe
    df_reduced = df[variables]
    st.dataframe(df_reduced.head(15).round(2))

    # data dimension
    st.write( 'Number of results found:', df.shape[0] )

    return df

def display_price_avg(df):
    try:
        df.round({'price': 2})
        #st.write( 'Average price: $ {:.2f}'.format(df.price.mean()) )
        st.write( 'Average price: ', round(df.price.mean(),2) )
    except:
        st.write( 'Error in obtaining price data.' )

    return None

def display_sqft_living_avg(df):
    try:
        df.round({'price': 2})
        st.write( 'Average living room area: ', round(df.sqft_living.mean(),2), 'sqft' )
    except:
        st.write( 'Error in obtaining area data.' )

    return None

def display_price_sqft(df):
    try:
        st.write( 'Average price per square foot: ', round(df.price_sqft.mean(),2) )
    except:
        st.write( 'Error in obtaining price data.' )

    return None

def display_stats(df, variables):
    st.header( 'Common statistics' )
    st.write( 'Statistical info:' )

    st.dataframe( df[variables].describe().applymap('{:,.2f}'.format) )

    return None

def display_stats_by_zipcode(df):

    # Average metrics
    df1 = df[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = df[['price', 'zipcode']].groupby('zipcode').median().reset_index()
    df4 = df[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df5 = df[['price_sqft', 'zipcode']].groupby('zipcode').mean().reset_index()
    df5 = df[['price_sqft', 'zipcode']].groupby('zipcode').mean().reset_index()
    df6 = df[df['season']=='summer'][['price','zipcode']].groupby('zipcode').median().reset_index()
    df7 = df[df['season']=='fall'][['price','zipcode']].groupby('zipcode').median().reset_index()
    df8 = df[df['season']=='winter'][['price','zipcode']].groupby('zipcode').median().reset_index()
    df9 = df[df['season']=='spring'][['price','zipcode']].groupby('zipcode').median().reset_index()

    # merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    m3 = pd.merge(m2, df4, on='zipcode', how='inner')
    m4 = pd.merge(m3, df5, on='zipcode', how='inner')
    m5 = pd.merge(m4, df6, on='zipcode', how='inner')
    m6 = pd.merge(m5, df7, on='zipcode', how='inner')
    m7 = pd.merge(m6, df8, on='zipcode', how='inner')
    df = pd.merge(m7, df9, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE AVG', 'PRICE MEDIAN', 'SQRT LIVING', 'PRICE/M2', 'PRICE MEDIAN SUMMER', 'PRICE MEDIAN FALL', 'PRICE MEDIAN WINTER', 'PRICE MEDIAN SPRING']

    st.header('Descriptive summary - by region and season')
    st.dataframe(df)

    return None

def define_price_levels(df, t1=321950, t2=450000, t3=645000):
    '''
    Define a price category/level based on thresholds
    From 0 to 3, where 3 is the most expensive
    '''
    if(len(df) == 0):
            st.write('No results to display.')
    else:

        for i in range(len(df)):
            if df.loc[i, 'price'] <= t1:
                df.loc[i, 'level'] = 0
            elif (df.loc[i,'price'] > t1) & (df.loc[i,'price'] <= t2):
                df.loc[i, 'level'] = 1
            elif (df.loc[i,'price'] > t2) & (df.loc[i,'price'] <= t3):
                df.loc[i, 'level'] = 2
            else:
                df.loc[i, 'level'] = 3

        df["level"] = df["level"].astype(str)

    return df

def display_commercial(df):
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    if(len(df) == 0):
        st.write('No results to display.')
    else:

        # Average price per year built
        st.sidebar.subheader('Select Max Year Built')
        min_year_built = int(df['yr_built'].min())
        max_year_built = int(df['yr_built'].max())
        if min_year_built == max_year_built: # if df has only ONE RESULT
            f_year_built = max_year_built
        else:
            f_year_built = st.sidebar.slider( 'Year Built', min_year_built, max_year_built, min_year_built )
            
        st.header('Average price per year built')
        # get date
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df2 = df.loc[df['yr_built'] <= f_year_built]
        df2 = df2[['yr_built', 'price']].groupby( 'yr_built' ).mean().reset_index()
        fig = px.line( df2, x='yr_built', y='price' )
        st.plotly_chart( fig, use_container_width=True )

        # Average price per day
        st.header( 'Average Price per day' )
        st.sidebar.subheader( 'Select Max Date' )
        min_date = datetime.strptime( df['date'].min(), '%Y-%m-%d' )
        max_date = datetime.strptime( df['date'].max(), '%Y-%m-%d' )
        if min_date == max_date: # if df has only ONE RESULT
            f_date = min_date
        else:
            f_date = st.sidebar.slider( 'Date', min_date, max_date, min_date )

        df['date'] = pd.to_datetime( df['date'] )
        df2 = df[df['date'] <= f_date]
        df2 = df2[['date', 'price']].groupby( 'date' ).mean().reset_index()

        fig = px.line( df2, x='date', y='price' )
        st.plotly_chart( fig, use_container_width=True )


        # Histogram
        st.header( 'Price Distribuition' )
        st.sidebar.subheader( 'Select Max Price' )
        min_price = int( df['price'].min() )
        max_price = int( df['price'].max() )
        avg_price = int( df['price'].mean() )

        if min_price == max_price: # if df has only ONE RESULT
            f_price = max_price
        else:
            f_price = st.sidebar.slider( 'Price', min_price, max_price, avg_price )

        df = df[df['price'] <= f_price]

        fig = px.histogram( df, x='price', nbins=50 )
        st.plotly_chart( fig, use_container_width=True )

    return None

def define_seasonality(df):

    df['month_day'] = pd.to_datetime(df['date']).dt.strftime('%m-%d')
    for i in range(len(df)):
        if '03-01' <= df.loc[i, 'month_day'] <= '05-31':
            df.loc[i, 'season'] = 'spring'
        elif '06-01' <= df.loc[i, 'month_day'] <= '08-31':
            df.loc[i, 'season'] = 'summer'
        elif '09-01' <= df.loc[i, 'month_day'] <= '11-30':
            df.loc[i, 'season'] = 'fall'
        else:
            df.loc[i, 'season'] = 'winter'

    return df

def define_price_median_per_region(df):

    # grouping by region and calculating median
    df1 = df[['price', 'zipcode']].groupby('zipcode')['price'].median().reset_index()
    df2 = df[df['season']=='summer'][['price','zipcode']].groupby('zipcode').median().reset_index()
    df3 = df[df['season']=='fall'][['price','zipcode']].groupby('zipcode').median().reset_index()
    df4 = df[df['season']=='winter'][['price','zipcode']].groupby('zipcode').median().reset_index()
    df5 = df[df['season']=='spring'][['price','zipcode']].groupby('zipcode').median().reset_index()

    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    m3 = pd.merge(m2, df4, on='zipcode', how='inner')
    data = pd.merge(m3, df5, on='zipcode', how='inner')
    data.columns = ['ZIP', 'PRICE', 'PRICE MEDIAN SUMMER', 'PRICE MEDIAN FALL', 'PRICE MEDIAN WINTER', 'PRICE MEDIAN SPRING']


    # setting price median on original df
    values = []
    values_summer = []
    values_fall = []
    values_winter = []
    values_spring = []
    for index, row in df.iterrows():
        values.append(data.loc[data['ZIP'] == row['zipcode'], 'PRICE'].values[0])
        values_summer.append(data.loc[data['ZIP'] == row['zipcode'], 'PRICE MEDIAN SUMMER'].values[0])
        values_fall.append(data.loc[data['ZIP'] == row['zipcode'], 'PRICE MEDIAN FALL'].values[0])
        values_winter.append(data.loc[data['ZIP'] == row['zipcode'], 'PRICE MEDIAN WINTER'].values[0])
        values_spring.append(data.loc[data['ZIP'] == row['zipcode'], 'PRICE MEDIAN SPRING'].values[0])
    df['price_med'] = values
    df['price_med_summer'] = values_summer
    df['price_med_fall'] = values_fall
    df['price_med_winter'] = values_winter
    df['price_med_spring'] = values_spring



    return df

def display_buy_sell_options(df, option, c):

    # determine if deal is good/it is worth to buy
    values = []
    for index, row in df.iterrows():
        if (row['price'] <= row['price_med']) & (row['condition'] >= 3):
            values.append('Yes')
        else:
            values.append('No')
    df['buy'] = values

    if option == 'rec':
        df_buy = df[df['buy']=='Yes']
    else:
        df_buy = df[df['buy']=='No']

    # determine ideal sell price based on seasonality and median per region
    values = []
    for index, row in df_buy.iterrows():
        if (row['season']=='summer'):
            if (row['price'] <= row['price_med_summer']):
                values.append(row['price']*1.3)
            else:
                values.append(row['price']*1.1)
        if (row['season']=='fall'):
            if (row['price'] <= row['price_med_fall']):
                values.append(row['price']*1.3)
            else:
                values.append(row['price']*1.1)
        if (row['season']=='winter'):
            if (row['price'] <= row['price_med_winter']):
                values.append(row['price']*1.3)
            else:
                values.append(row['price']*1.1)
        if (row['season']=='spring'):
            if (row['price'] <= row['price_med_spring']):
                values.append(row['price']*1.3)
            else:
                values.append(row['price']*1.1)

    df_buy['sell_price'] = values
    df_buy['revenue'] = df_buy['sell_price'] - df_buy['price']
    revenue = str(round(df_buy['revenue'].sum(), 2))
    revenue_avg = str(round(df_buy['revenue'].mean(), 2))

    st.dataframe(df_buy[['id','zipcode','price','season','condition','price_med','price_med_summer','price_med_fall','price_med_winter','price_med_spring','buy','sell_price','revenue']])
    if option == 'rec':
        st.write('Total revenue: $ ', revenue)
        st.write('Average revenue per property: $ ', revenue_avg)

    return df_buy

def display_price_map(df, geofile):
    st.title('Region Overview')

    c1, c2 = st.columns(2, gap='large')
    c1.header('Portfolio Density')

    df2 = df.copy()

    if(len(df) == 0):
        st.write('No results to display.')
    else:

        # Base Map - Folium
        density_map = folium.Map(location=[df['lat'].mean(), df['long'].mean()],
                                default_zoom_start=15) 

        marker_cluster = MarkerCluster().add_to(density_map)
        for name, row in df2.iterrows():
            folium.Marker([row['lat'], row['long']], 
                popup='Sold ${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                    row['price'], 
                    row['date'], 
                    row['sqft_living'],
                    row['bedrooms'],
                    row['bathrooms'],
                    row['yr_built'])
                ).add_to(marker_cluster)


        with c1:
            folium_static(density_map)


        # Region Price Map
        c2.header('Price Density')

        df2 = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
        df2.columns = ['ZIP', 'PRICE']

        geofile = geofile[geofile['ZIP'].isin(df2['ZIP'].tolist())]

        region_price_map = folium.Map( 
            location=[df['lat'].mean(), 
            df['long'].mean() ],
            default_zoom_start=15
        ) 


        region_price_map.choropleth(data = df2,
                                    geo_data = geofile,
                                    columns=['ZIP', 'PRICE'],
                                    key_on='feature.properties.ZIP',
                                    fill_color='YlOrRd',
                                    fill_opacity = 0.7,
                                    line_opacity = 0.2,
                                    legend_name='AVG PRICE')

        with c2:
            folium_static(region_price_map)


    return None

def create_price_slider(min, max, med, c):

    price_slider = st.slider('Price Range', min, max, med)

    return price_slider

def display_map(df, c1, c2):

    # slider filter
    if int(df['price'].min()) == int(df['price'].max()): # if df has only ONE RESULT
        price_slider = int(df['price'].max())
    else:
        price_slider = create_price_slider( min=int(df['price'].min()), max=int(df['price'].max()), med=int(df['price'].median()), c=c1 )

    # select rows
    houses = df[df['price'] <= price_slider][['id','lat','long',
                                                'price', 'level', 'zipcode']]
    # draw map
    fig = px.scatter_mapbox( 
        houses, 
        lat="lat", 
        lon="long", 
        color="level", 
        size="price",
        color_discrete_sequence=px.colors.qualitative.D3,
        size_max=15, 
        zoom=10,
        hover_data=["zipcode", "id", "price"],
        )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart( fig )

    return None

def display_h1(df, c):

    df1 = df[df['waterfront'] == 1]
    df2 = df[df['waterfront'] == 0]
    s1 = pd.Series([df1["price"].median(), df2["price"].median()], name="Price median")
    s2 = pd.Series(['Waterfront', 'No Waterfront'], name="Type")

    dframe = pd.DataFrame({ 'Price median': s1, 'Type': s2 })

    fig = px.bar(data_frame=dframe, x='Type', y='Price median')
    pct = round(100*(dframe[dframe['Type']=='Waterfront']['Price median'].values[0] / dframe[dframe['Type']=='No Waterfront']['Price median'].values[0] ), 2)
    pct = str(pct) + '%'

    c.markdown(
        """
        :x: **FALSE:** It is noticed by the chart that waterfront houses are in fact **311%** more expensive than the ones which are not.
        """
    )
    c.plotly_chart( fig, use_container_width=True )
    c.write(f'Percentage between prices: {pct}')

    return None

def display_h2(df, c):

    df1 = df[(df['waterfront']==1) & ((df.season.isin(['winter','fall'])))]
    df2 = df[(df['waterfront']==1) & ((df.season.isin(['summer','spring'])))]
    s1 = pd.Series([df1.price.mean(), df2.price.mean()], name="Price median")
    s2 = pd.Series(['Winter/Fall', 'Summer/Spring'], name="Type")

    dframe = pd.DataFrame({ 'Price median': s1, 'Type': s2 })

    fig = px.bar(data_frame=dframe, x='Type', y='Price median')
    pct = round(100 - 100*(dframe[dframe['Type']=='Winter/Fall']['Price median'].values[0] / dframe[dframe['Type']=='Summer/Spring']['Price median'].values[0] ), 2)
    pct = str(pct) + '%'

    c.markdown(
        """
        :heavy_check_mark: **TRUE:** We can see that houses are announced about **8.3%** cheaper in winter/fall in comparison to summer/spring.
        """
    )
    c.plotly_chart( fig, use_container_width=True )
    c.write(f'Percentage between prices: {pct}')

    return None

def display_h3(df, c):

    df1 = df[df['sqft_basement'] != 0]
    df2 = df[df['sqft_basement'] == 0]
    s1 = pd.Series([df1["price"].median(), df2["price"].median()], name="Price median")
    s2 = pd.Series(['With Basement', 'No Basement'], name="Type")

    dframe = pd.DataFrame({ 'Price median': s1, 'Type': s2 })

    fig = px.bar(data_frame=dframe, x='Type', y='Price median')
    pct = round(100*(dframe[dframe['Type']=='With Basement']['Price median'].values[0] / dframe[dframe['Type']=='No Basement']['Price median'].values[0] ), 2)
    pct = str(pct) + '%'

    c.markdown(
        """
        :x: **FALSE:** Houses with basement are actually **25%** more expensive than houses without them.
        """
    )
    c.plotly_chart( fig, use_container_width=True )
    c.write(f'Percentage between prices: {pct}')

    return None

def display_h4(df, c):

    # creating groupby by price median
    df['year_month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    df1 = df[['price', 'year_month']].groupby('year_month')['price'].median().reset_index()
    df1.columns = ['MONTH', 'PRICE MED']

    # creating column of monthly increase
    values = []
    values.append(0)

    for index, row in df1.iterrows():
        if index != 0:
            values.append( ((df1.loc[index,'PRICE MED'] - df1.loc[index-1,'PRICE MED']))/df1.loc[index-1,'PRICE MED']*100 )
            
    df1['diff_month'] = values
    df1['diff_month'] = df1['diff_month'].apply(lambda x: str(round(x, 2)) + "%")
    #c.dataframe(df1)


    fig = px.line(data_frame=df1, x='MONTH', y='PRICE MED', text='diff_month')

    c.markdown(
        """
        :heavy_check_mark: **TRUE:** As we can observe the highest increase in price percentage in the observed period is 5.89%, from March to April 2015.
        """
    )
    c.plotly_chart( fig, use_container_width=True )

    return None

def display_h5(df):

    df1 = df[(df['yr_built'] < 1955) & (df['yr_renovated'] != 0)]
    df2 = df[df['yr_built'] >= 1955]
    s1 = pd.Series([df1["price"].median(), df2["price"].median()], name="Price median")
    s2 = pd.Series(['Earlier than 1955 and renovated', 'Other'], name="Type")

    dframe = pd.DataFrame({ 'Price median': s1, 'Type': s2 })

    fig = px.bar(data_frame=dframe, x='Type', y='Price median')
    pct = round(100*(dframe[dframe['Type']=='Earlier than 1955 and renovated']['Price median'].values[0] / dframe[dframe['Type']=='Other']['Price median'].values[0] ), 2)
    pct = str(pct) + '%'

    st.markdown(
        """
        :heavy_check_mark: **TRUE:** Properties that were built prior to 1955 and which have been renovated are priced **32%** higher than those that haven't or those that were built at a later time.
        """
    )
    st.plotly_chart( fig, use_container_width=True )
    st.write(f'Percentage between prices: {pct}')

    return None

# load data
if __name__ == "__main__":
    try:
        ### 1. INTRODUCTION ###
        c1, c2, c3 = st.columns(3)
        icon= 'https://github.com/VictorBSR/Data_Science/blob/main/house_rocket/icon.png?raw=true'
        c2.image(icon, caption= 'logo', width=350)
        st.markdown("<h1 style='text-align: center; color: black;'>House Rocket - Dashboard</h1>", unsafe_allow_html=True)
        st.markdown( '### House Prices Data Analysis' )
        st.write('*House Rocket* is a fictional company which deals with real estate operations (purchase and sales of properties) in the city of Seattle/USA. The main goal of this page is to provide useful insights, graphs, maps and a general overview of the houses data in order to aid quick and effective decision making by the board of directors, posing as a modern and smart approach instead of manual analysis of the large amount of data provided.')

        ### 2. BUSINESS QUESTIONS ###
        st.markdown(
        """
        **Main points to be covered:**
        - **Which properties should the company buy?**
        - **At which price should the company sell the acquired properties?**
        - **When is the best time to sell the acquired properties?**
        """
        )

        ### 3. DATA OVERVIEW ###

        # load data
        path = 'kc_house_data.csv'
        url='https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
        data, full_data = get_data( path )
        geofile = get_geofile( url )

        # transform data with filters
        (variables, zipcodes, bedrooms, bathrooms, floors, waterfront, condition, renovated) = define_filters(data)
        
        data.reset_index(drop=True, inplace=True)
        full_data.reset_index(drop=True, inplace=True)
        data = define_seasonality(data)
        data = define_price_median_per_region(data)
        full_data = define_seasonality(full_data)
        full_data = define_price_median_per_region(full_data)

        df = set_filters(data, zipcodes, bedrooms, bathrooms, floors, waterfront, condition, renovated)
        full_df = set_filters(full_data, zipcodes, bedrooms, bathrooms, floors, waterfront, condition, renovated)

        # return views
        st.header( 'General overview' )
        disp = st.checkbox('Display sample dataset')
        if disp:
            # dataframes, tables and infos
            df = display_basic_info(df, variables)
            display_price_avg(df)
            display_sqft_living_avg(df)
            display_price_sqft(df)

        ### 5. ASSUMPTIONS ###
        st.header( 'Assumptions' )
        st.markdown(
        """
        - **Houses with over 11 bedrooms are considered to be outliers and thus are ignored.**
        - **Only the most recent data for a given property is consiered in the overview and map views, in case it has multiple entries.**
        - **Values equal to 0 (zero) mean the feature is not applicable (e.g. does not have waterfront, was not renovated, does not have basement, and so on.**
        """
        )


        ### 4. BUSINESS HYPOTHESES ###
        st.header( 'Business Hypotheses' )
        disp_bh = st.checkbox('Display business insights')
        if disp_bh:
            c1,c2 = st.columns((2))

            c1.markdown(
                """
                ### :fast_forward: Waterfront houses are 25% more expensive than average of houses with no waterfront
                """
            )
            display_h1(data, c1)


            c2.markdown(
                """
                ### :fast_forward: Houses with waterfront are about 10% cheaper when sold on winter or fall
                """
            )
            display_h2(data, c2)


            c1.markdown(
                """
                ### :fast_forward: Having a basement makes properties circa 50% more expensive
                """
            )
            display_h3(data, c1)


            c2.markdown(
                """
                ### :fast_forward: The highest Month-over-Month increase in property prices is over 5%
                """
            )
            display_h4(full_data, c2)


            c1.markdown(
                """
                ### :fast_forward: Properties with built date earlier than 1955 which have been renovated boast prices over 20% above average in comparison to others.
                """
            )
            display_h5(data)

        ### 5. MAP VIEW AND PURCHASE RECOMMENDATIONS ###
        st.header('Map view and purchase recommendations')
        disp_map = st.checkbox('Display map view and recommendations')
        if disp_map:
            display_stats_by_zipcode(df)

            # maps
            display_price_map(df, geofile)
            df = define_price_levels(df, t1=321950, t2=450000, t3=645000)

            # buy/sell analysis
            st.title('Purchase viability - Business Insights')
            st.write('Criteria: if the price of a given house is equal to or lower than the price median for the region (zipcode) and the condition is good (3) or higher, the house is a viable deal and a possible option to be bought. Sell price is recommended to be circa 30% higher than purchase price if the latter is below the median for that given season, or 10% higher otherwise.')

            c1,c2 = st.columns((2))
            disp_recommended = st.checkbox('Display recommended deals')
            if disp_recommended:
                st.header('Recommended deals:')
                df_buy = display_buy_sell_options(df, 'rec', c1)
                display_map(df_buy, c1, c2)
            disp_not_recommended = st.checkbox('Display not recommended deals')
            if disp_not_recommended:
                st.header('Not recommended deals:')
                df_not_buy = display_buy_sell_options(df, 'not rec', c1)
                display_map(df_not_buy, c1, c2)


        st.markdown("*Made by: Victor Reis* :coffee:")
    except Exception as ex:
        st.write('An exception occured. Please try again or inform the owner of this project.')
        st.write(ex)
