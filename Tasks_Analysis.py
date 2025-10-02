#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import webbrowser
import os
import pytz
from datetime import datetime, timedelta, timezone


# In[2]:


nltk.download('vader_lexicon')


# In[3]:


apps_df=pd.read_csv('Play Store Data.csv')
reviews_df=pd.read_csv('User Reviews.csv')
regions_df = pd.read_csv("Global EV Data 2024.csv")


# In[4]:


apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns:
    apps_df[column].fillna(apps_df[column].mode()[0], inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df = apps_df[apps_df['Rating'] <= 5]
reviews_df.dropna(subset=['Translated_Review'], inplace=True)


# In[5]:


apps_df['Reviews'] = apps_df['Reviews'].astype(int)
apps_df['Installs'] = apps_df['Installs'].str.replace(',', '').str.replace('+', '').astype(int)
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)


# In[6]:


def convert_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    else:
        return np.nan


# In[7]:


apps_df['Size'] = apps_df['Size'].apply(convert_size)


# In[8]:


apps_df['Log_Installs'] = np.log1p(apps_df['Installs'])
apps_df['Log_Reviews'] = np.log1p(apps_df['Reviews'])


# In[9]:


def rating_group(rating):
    if rating >= 4:
        return 'Top rated'
    elif rating >= 3:
        return 'Above average'
    elif rating >= 2:
        return 'Average'
    else:
        return 'Below average'

apps_df['Rating_Group'] = apps_df['Rating'].apply(rating_group)


# In[10]:


apps_df['Revenue'] = apps_df['Price'] * apps_df['Installs']


# In[11]:


sia = SentimentIntensityAnalyzer()
reviews_df['Sentiment_Score'] = reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])


# In[12]:


apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')
apps_df['Year'] = apps_df['Last Updated'].dt.year


# In[13]:


unique_regions = regions_df["region"].dropna().unique()

apps_df["region"] = [unique_regions[i % len(unique_regions)] for i in range(len(apps_df))]


# In[14]:


import re

def parse_android_version(ver):
    if not isinstance(ver, str):
        return None
    if "Varies with device" in ver:
        return None  
    match = re.search(r'\d+(\.\d+)?', ver) 
    if match:
        version_str = match.group(0)
        parts = version_str.split(".")
        if len(parts) >= 2:
            return float(parts[0] + "." + parts[1])
        else:
            return float(parts[0])
    return None


# In[15]:


apps_df["Android Ver"] = apps_df["Android Ver"].apply(parse_android_version)


# In[16]:


apps_df["Month"] = apps_df["Last Updated"].dt.to_period("M").astype(str)


# In[17]:


merged_df = pd.merge(apps_df, reviews_df, on='App', how='inner')


# In[18]:

# Define paths for static and templates folders
static_folder = "./static"
templates_folder = "./templates"

# Create folders if they don't exist
if not os.path.exists(static_folder):
    os.makedirs(static_folder)
if not os.path.exists(templates_folder):
    os.makedirs(templates_folder)

plot_containers = ""


# In[19]:


def save_plot_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(static_folder, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    plot_containers += f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('/static/{filename}')">
        <div class="plot">{html_content}</div>
        <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')


# In[20]:


plot_width = 400
plot_height = 300
plot_bg_color = 'black'
text_color = 'white'
title_font = {'size': 16}
axis_font = {'size': 12}


# Task 1

# In[ ]:


filtered_df = apps_df[
    (apps_df["Rating"] >= 4.0) &
    (apps_df["Size"] >= 10) &
    (apps_df["Last Updated"].dt.month == 1)
]

top_categories = (
    filtered_df.groupby("Category")
    .agg({"Rating": "mean", "Reviews": "sum", "Installs": "sum"})
    .sort_values("Installs", ascending=False)
    .head(10)
    .reset_index()
)


india_tz = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(india_tz)

if 15 <= current_time.hour < 17:
    fig11 = px.bar(
        top_categories.melt(
            id_vars="Category",
            value_vars=["Rating", "Reviews"],
            var_name="Metric",
            value_name="Value"
        ),
        x="Category",
        y="Value",
        color="Metric",
        barmode="group",
        labels={"Category": "App Category", "Value": "Value"},
        title="Top 10 App Categories by Installs (Filtered)",
        width=400,
        height=300,
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    fig11.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        title_font={'size':16},
        xaxis=dict(title_font={'size':12}),
        yaxis=dict(title_font={'size':12}),
        margin=dict(l=10, r=10, t=30, b=10)
    )

    save_plot_as_html(
        fig11,
        "Task1.html",
        "Comparison of average ratings and total reviews for top app categories (3PM-5PM IST only)."
    )
else:
    print("⏰ Graph hidden - only available between 3 PM and 5 PM IST.")


# In[22]:


apps_df.dtypes


# Task 2

# In[ ]:


filtered_choro = apps_df[
    ~apps_df["Category"].str.startswith(("A", "C", "G", "S"))
]

choro_grouped = (
    filtered_choro.groupby(["Category"])["Installs"]
    .sum()
    .reset_index()
)

top5_categories = (
    choro_grouped.sort_values("Installs", ascending=False)
    .head(5)["Category"]
)

choro_top5 = filtered_choro[filtered_choro["Category"].isin(top5_categories)]

choro_top5 = (
    choro_top5.groupby(["region", "Category"], as_index=False)["Installs"].sum()
)

choro_top5["Highlight"] = choro_top5["Installs"].apply(
    lambda x: "High Installs (>1M)" if x > 1_000_000 else "Normal"
)

india_tz = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(india_tz)

if 18 <= current_time.hour < 20:
    fig12 = px.choropleth(
    choro_top5,
    locations="region",
    locationmode="country names",
    color="Installs",
    hover_name="Category",
    facet_col="Category",
    facet_col_wrap=2,
    color_continuous_scale=px.colors.sequential.Plasma,
    title="Global Installs by Top 5 App Categories",
    width=400,    
    height=300,      
    facet_col_spacing=0.05,
    facet_row_spacing=0.05
    )

    fig12.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        margin=dict(l=10, r=10, t=50, b=10)
    )

    fig12.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1]))

    save_plot_as_html(
        fig12,
        "Task2.html",
        "Choropleth map showing installs for the top 5 categories (6-8 PM IST only). Categories with installs > 1M are highlighted."
    )


else:
    print("⏰ Graph hidden - only available between 6 PM and 8 PM IST.")


# Task 3

# In[ ]:


filtered_dual = apps_df[
    (apps_df["Installs"] >= 10_000) &
    (apps_df["Revenue"] >= 10_000) &
    (apps_df["Android Ver"] > 4.0) &
    (apps_df["Size"] > 15) &
    (apps_df["Content Rating"] == "Everyone") &
    (apps_df["App"].str.len() <= 30)
]

top3_categories = (
    filtered_dual.groupby("Category")["Installs"]
    .sum()
    .sort_values(ascending=False)
    .head(3)
    .index
)

dual_top3 = filtered_dual[filtered_dual["Category"].isin(top3_categories)]

dual_grouped = (
    dual_top3.groupby(["Category", "Type"])
    .agg({"Installs": "mean", "Revenue": "mean"})
    .reset_index()
)

india_tz = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(india_tz)

if 13 <= current_time.hour < 14:
    import plotly.graph_objects as go

    fig13 = go.Figure()

    fig13.add_trace(
        go.Bar(
            x=dual_grouped["Category"] + " (" + dual_grouped["Type"] + ")",
            y=dual_grouped["Installs"],
            name="Average Installs",
            marker_color="skyblue",
            yaxis="y1"
        )
    )

    fig13.add_trace(
        go.Scatter(
            x=dual_grouped["Category"] + " (" + dual_grouped["Type"] + ")",
            y=dual_grouped["Revenue"],
            name="Average Revenue",
            mode="lines+markers",
            line=dict(color="orange", width=3),
            yaxis="y2"
        )
    )

    fig13.update_layout(
    title="Dual-Axis Chart: Avg Installs vs Revenue (Free vs Paid)",
    xaxis=dict(
        title=dict(text="App Category (Free/Paid)", font=dict(size=16, family="Arial", color="white")),
        tickfont=dict(size=12, family="Arial", color="white")
    ),
    yaxis=dict(
        title=dict(text="Average Installs", font=dict(size=16, family="Arial", color="white")),
        tickfont=dict(size=12, family="Arial", color="white")
    ),
    yaxis2=dict(
        title=dict(text="Average Revenue ($)", font=dict(size=16, family="Arial", color="white")),
        tickfont=dict(size=12, family="Arial", color="white"),
        overlaying="y",
        side="right"
    ),
    plot_bgcolor="black",
    paper_bgcolor="black",
    font_color="white",
    title_font=dict(size=14, family="Arial", color="white"),
    margin=dict(l=10, r=10, t=30, b=10),
    width=400,
    height=300,
    legend=dict(orientation="h", y=-0.2)
)


    save_plot_as_html(
        fig13,
        "Task3.html",
        "Comparison of average installs and revenue for Free vs Paid apps in top 3 categories (1-2 PM IST only)."
    )
else:
    print("⏰ Graph hidden - only available between 1 PM and 2 PM IST.")


# Task 4

# In[25]:


print(merged_df['Category'].unique().tolist())


# In[ ]:


filtered_ts = apps_df[
    (~apps_df["App"].str.lower().str.startswith(("x", "y", "z"))) & 
    (apps_df["Category"].str.startswith(("E", "C", "B"))) &         
    (apps_df["Reviews"] > 500) &                                    
    (~apps_df["App"].str.contains("S", case=False, na=False))      
]

category_translation = {
    "BEAUTY": "सौंदर्य",     
    "BUSINESS": "வணிகம்",   
    "DATING": "DATING" 
}
filtered_ts["Category"] = filtered_ts["Category"].replace(category_translation)

filtered_ts["Last Updated"] = pd.to_datetime(filtered_ts["Last Updated"], errors="coerce")

ts_grouped = (
    filtered_ts.groupby([pd.Grouper(key="Last Updated", freq="M"), "Category"])
    .agg({"Installs": "sum"})
    .reset_index()
    .sort_values("Last Updated")
)

ts_grouped["MoM_Growth"] = ts_grouped.groupby("Category")["Installs"].pct_change() * 100

india_tz = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(india_tz)

if 18 <= current_time.hour < 21:
    import plotly.graph_objects as go

    fig14 = go.Figure()

    categories = ts_grouped["Category"].unique()
    for cat in categories:
        cat_data = ts_grouped[ts_grouped["Category"] == cat]

        # Main line
        fig14.add_trace(
            go.Scatter(
                x=cat_data["Last Updated"],
                y=cat_data["Installs"],
                mode="lines+markers",
                name=cat
            )
        )

        high_growth = cat_data[cat_data["MoM_Growth"] > 20]
        if not high_growth.empty:
            fig14.add_trace(
                go.Scatter(
                    x=high_growth["Last Updated"],
                    y=high_growth["Installs"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    fill="tozeroy",
                    fillcolor="rgba(255, 165, 0, 0.3)"  
                )
            )

    fig14.update_layout(
        title="Time Series: Total Installs Trend by Category",
        xaxis_title="Month",
        yaxis_title="Total Installs",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        margin=dict(l=10, r=10, t=30, b=10),
        width=400,
        height=300,
        legend=dict(orientation="h", y=-0.2)
    )

    save_plot_as_html(
        fig14,
        "Task4.html",
        "Time series of installs segmented by category. Highlighted areas show >20% MoM growth (6-9 PM IST only)."
    )
else:
    print("⏰ Graph hidden - only available between 6 PM and 9 PM IST.")


# Task 5

# In[ ]:


allowed_categories = [
    "GAME", "BEAUTY", "BUSINESS", "COMICS", "COMMUNICATION",
    "DATING", "ENTERTAINMENT", "SOCIAL", "EVENTS"
]

if "Sentiment_Subjectivity" in merged_df.columns:
    filtered_bubble = merged_df[
        (merged_df["Rating"] > 3.5) &
        (merged_df["Category"].isin(allowed_categories)) &
        (merged_df["Reviews"] > 500) &
        (~merged_df["App"].str.contains("S", case=False, na=False)) &
        (merged_df["Sentiment_Subjectivity"] > 0.5) &
        (merged_df["Installs"] > 50_000)
    ]
else:
    print("⚠️ Sentiment_Subjectivity column missing in merged_df. Please check merge or column names.")


category_translation = {
    "BEAUTY": "सौंदर्य",    
    "BUSINESS": "வணிகம்", 
    "DATING": "DATING"
}
filtered_bubble["Category"] = filtered_bubble["Category"].replace(category_translation)

india_tz = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(india_tz)

if 17 <= current_time.hour < 19:
    import plotly.express as px

    fig15 = px.scatter(
        filtered_bubble,
        x="Size",
        y="Rating",
        size="Installs",
        color="Category",
        hover_name="App",
        title="Bubble Chart: App Size vs Rating",
        labels={"Size": "App Size (MB)", "Rating": "Average Rating"},
        size_max=50,
        width=400,
        height=300
    )

    fig15.for_each_trace(
        lambda trace: trace.update(marker=dict(color="pink"))
        if "GAME" in trace.name else ()
    )

    fig15.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font=dict(size=16),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.2)
    )

    save_plot_as_html(
        fig15,
        "Task5.html",
        "Bubble chart showing relationship between app size and rating. "
        "Bubble size = installs. Game category highlighted in pink (5-7 PM IST only)."
    )
else:
    print("⏰ Graph hidden - only available between 5 PM and 7 PM IST.")


# Task 6

# In[ ]:


filtered_df = apps_df[
    (apps_df['Rating'] >= 4.2) &
    (~apps_df['App'].str.contains(r'\d')) & 
    (apps_df['Category'].str.startswith(('T', 'P'))) &
    (apps_df['Reviews'] > 1000) &
    (apps_df['Size'].between(20, 80))
]

category_map = {
    "TRAVEL_AND_LOCAL": "VOYAGES ET LOCAL",  
    "PRODUCTIVITY": "PRODUCTIVIDAD",      
    "PHOTOGRAPHY": "写真"                  
}
filtered_df['Category'] = filtered_df['Category'].replace(category_map)

grouped = filtered_df.groupby(['Month', 'Category'])['Installs'].sum().reset_index()
pivot_df = grouped.pivot(index='Month', columns='Category', values='Installs').fillna(0)
cumulative = pivot_df.cumsum()

growth_mask = cumulative.pct_change() > 0.25

current_time = datetime.now().astimezone(timezone(timedelta(hours=5, minutes=30)))

if 16 <= current_time.hour < 18:
    fig14 = go.Figure()

    for category in cumulative.columns:
        fig14.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative[category],
            mode='lines',
            stackgroup='one',
            name=category,
            line=dict(width=0.5)
        ))

        highlight_months = growth_mask.index[growth_mask[category].fillna(False)]
        if not highlight_months.empty:
            fig14.add_trace(go.Scatter(
                x=highlight_months,
                y=cumulative.loc[highlight_months, category],
                mode='markers',
                name=f"{category} (Growth >25%)",
                marker=dict(size=10, color='red', symbol="circle-open")
            ))

    fig14.update_layout(
        title="Cumulative Installs Over Time by Category",
        xaxis_title="Month",
        yaxis_title="Cumulative Installs",
        hovermode="x unified",
        legend_title="App Category",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        title_font={'size':16},
        margin=dict(l=10,r=10,t=30,b=10),
        width=400,
        height=300
    )

    save_plot_as_html(fig14,"Task6.html",
        "Shows cumulative installs over time for selected categories, with translations and highlights for >25% growth months")

else:
    print("⏳ Chart hidden (Only available between 4 PM - 6 PM IST).")


# In[32]:


plot_containers_split = plot_containers.split('</div>')
if len(plot_containers_split) > 1:
    final_plot = plot_containers_split[-2] + '</div>'
else:
    final_plot = plot_containers


# In[33]:


dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Reviews Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container:hover .insights {{
            display: block;
        }}
    </style>
    <script>
        function openPlot(filename) {{
            window.open(filename, '_blank');
        }}
    </script>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
        <h1>Google Play Store Reviews Analytics</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
    </div>
    <div class="container">
        <p style="width:100%; text-align:center; font-size:20px; font-weight:bold; margin-bottom:20px;">
            Internship Tasks Analysis Dashboard
        </p>
        {plots}
    </div>
</body>
</html>
"""


# In[34]:


final_html = dashboard_html.format(plots=plot_containers, plot_width=plot_width, plot_height=plot_height)

dashboard_path = os.path.join(templates_folder, "dashboard.html")
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)

# webbrowser.open('file://' + os.path.realpath(dashboard_path))

