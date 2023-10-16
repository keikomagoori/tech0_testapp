import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# Streamlitアプリのタイトル
st.title("営業ターゲット分析／検索システム")

# スクレイピング対象のターゲットURL 
#　実行の所要時間を考慮し、東東京、ワンルーム～２DKに絞り込み
base_url = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&ta=13&sc=13106&sc=13107&sc=13108&sc=13118&sc=13121&sc=13122&sc=13123&cb=0.0&ct=9999999&et=9999999&md=01&md=02&md=03&md=04&md=05&md=06&cn=9999999&mb=0&mt=9999999&shkr1=03&shkr2=03&shkr3=03&shkr4=03&fw2=&srch_navi=1"

# ページ数を指定
# total_pages = 10

# データを格納するためのリストを初期化
data = {
    '物件名': [],
    '住所': [],
    '築年数': [],
    '家賃': [],
    '間取り': [],
    '面積': [],
    'アクセス2': [],
    'アクセス3': [],
    'アクセス4': [],
    'アクセス5': [],
    'アクセス6': [],
    'アクセス7': [],
    'アクセス8': [],
    'アクセス9': [] 
}

# StreamlitのUI要素を追加
#サイドバー
place = st.sidebar.selectbox("エリア", ("", "台東区", "墨田区", "江東区", "荒川区", "足立区", "葛飾区", "江戸川区"))
total_pages = st.sidebar.selectbox("スクレイピングするページ数", list(range(1, 936)), index=9) # ページ数の選択
flag = st.sidebar.selectbox("フラグ選択", ("", "ターゲットフラグ", "サブターゲットフラグ"))
stock = st.sidebar.text_input("滞留期間（W）")
year = st.sidebar.text_input("築年数(年)")
access = st.sidebar.text_input("最短アクセス(~分以内)")
monthly_fee = st.sidebar.selectbox("家賃",("平均以下","平均以上","絞り込みなし"))
start_button = st.sidebar.button("検索")  # スクレイピングを開始するボタン

if st.sidebar.start_button:
    st.text("検索中...")

    # 各ページをスクレイピング
    for page_number in range(1, total_pages + 1):
        # ページ番号を含めたURLを生成s
        url = base_url + f"&page={page_number}"

        # URLからページを取得
        response = requests.get(url)

        # ページのHTMLを解析
        soup = BeautifulSoup(response.text, 'html.parser')

         # 物件名、住所、築年数、家賃、間取り、面積、アクセス情報を取得
        cassette_items = soup.select('div.cassetteitem')
        for item in cassette_items:
            # 物件名、住所、築年数、家賃、間取り、面積を取得
            title = item.select_one('.cassetteitem_content-title').text.strip()
            address = item.select_one('.cassetteitem_detail-col1').text.strip()
            chikunensu = item.select_one('.cassetteitem_detail-col3').text.strip()
            rent = item.select_one('span.cassetteitem_price--rent').text.strip()
            layout = item.select_one('.cassetteitem_madori').text.strip()
            area = item.select_one('.cassetteitem_menseki').text.strip()

            # アクセス情報を取得
            access_list = []
            access_elements = item.select('.cassetteitem_detail-text')
            for access_element in access_elements:
                access_list.append(access_element.text.strip())

            # データをdataに追加
            data['物件名'].append(title)
            data['住所'].append(address)
            data['築年数'].append(chikunensu)
            data['家賃'].append(rent)
            data['間取り'].append(layout)
            data['面積'].append(area)

            # アクセス情報を追加
            for j in range(8):  # 最大8つのアクセス情報まで対応
                column_name = f'アクセス{j+2}'
                if j < len(access_list):
                    data[column_name].append(access_list[j])
                else:
                    data[column_name].append('')  # アクセス情報がない場合は空白を挿入

        # スクレイピングの間隔を設定（ウェブサーバーに負荷をかけないために）
        time.sleep(1)

    # データをDataFrameに変換
    df = pd.DataFrame(data)

    # ダブルクォートを削除
    df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)

    # 欠損値を0で置き換える
    df = df.fillna('0')

    # アクセス列から数字のみを抽出して新しい列を作成
    def extract_minute(access):
        match = re.search(r'\d+', str(access))
        if match:
            return int(match.group())
        else:
            return None

    #以下のコード書き換え→元に戻す！
    #for j in range(2, 10):
    #    column_name = f'アクセス{j}'
    #    df[column_name] = df[column_name].apply(extract_minute)
    df['minute2'] = df['アクセス2'].apply(extract_minute)
    df['minute3'] = df['アクセス3'].apply(extract_minute)
    df['minute4'] = df['アクセス4'].apply(extract_minute)
    df['minute5'] = df['アクセス5'].apply(extract_minute)
    df['minute6'] = df['アクセス6'].apply(extract_minute)
    df['minute7'] = df['アクセス7'].apply(extract_minute)
    df['minute8'] = df['アクセス8'].apply(extract_minute)
    df['minute9'] = df['アクセス9'].apply(extract_minute)

    # 必要な列の順序を調整
    df = df[['物件名', '住所', '築年数', '家賃', '間取り', '面積', 'アクセス2', 'minute2', 'アクセス3', 'minute3', 'アクセス4', 'minute4', 'アクセス5', 'minute5', 'アクセス6', 'minute6', 'アクセス7', 'minute7', 'アクセス8', 'minute8', 'アクセス9', 'minute9']]

    #★★★データ分析
    
    # 0をNaNに変換 min_minuteに0が入らないようにする
    df[['minute2', 'minute3', 'minute4', 'minute5', 'minute6', 'minute7', 'minute8', 'minute9']] = df[['minute2', 'minute3', 'minute4', 'minute5', 'minute6', 'minute7', 'minute8', 'minute9']].replace(0, np.nan)


    # ★★最小値を抽出し新しい列 min_minute に追加
    df['min_minute'] = df[['minute2', 'minute3', 'minute4', 'minute5', 'minute6', 'minute7', 'minute8', 'minute9']].min(axis=1)

    # '築年数' 列を文字列型に変換
    df['築年数'] = df['築年数'].astype(str)

    # '家賃' 列と '面積' 列を数値型に変換
    df['家賃'] = df['家賃'].str.replace('万円', '').str.replace(',', '').astype(float)
    df['面積'] = df['面積'].str.replace('m2', '').astype(float)


    # 築年数2列を追加
    df['築年数2'] = df['築年数'].apply(lambda x: '築１年' if '新築' in x else x)

    # 築年数2列から数字のみを抽出して築年数_値列を追加
    df['築年数_値'] = df['築年数2'].apply(lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else None)

    # '家賃' 列を '面積' 列で割って '家賃/平米' 列を追加
    df['家賃/平米'] = df['家賃'] / df['面積']

    # 必要な列の順序を調整
    df = df[['物件名', '住所', '築年数', '築年数2', '築年数_値', '家賃', '間取り', '面積', '家賃/平米', 'アクセス2', 'minute2', 'アクセス3', 'minute3', 'アクセス4', 'minute4', 'アクセス5', 'minute5', 'アクセス6', 'minute6', 'アクセス7', 'minute7', 'アクセス8', 'minute8', 'アクセス9', 'minute9', 'min_minute']]

    # 築年数_値の平均値を計算
    average_age = df['築年数_値'].mean()

    # 家賃/平米の平均値を計算
    average_rent_per_sqm = df['家賃/平米'].mean()

    # min_minuteの平均値を計算
    average_min_minute = df['min_minute'].mean()

    # 築年数フラグの追加
    df['築年数フラグ'] = df.apply(lambda row: 1 if row['築年数_値'] > average_age else 0, axis=1)

    # 家賃フラグの追加
    df['家賃フラグ'] = df.apply(lambda row: 1 if row['家賃/平米'] < average_rent_per_sqm else 0, axis=1)

    # アクセスフラグの追加
    df['アクセスフラグ'] = df.apply(lambda row: 1 if row['min_minute'] > average_min_minute else 0, axis=1)

    # 築年数フラグ、家賃フラグ、アクセスフラグがすべて1の行にターゲットフラグを1とする
    df['ターゲットフラグ'] = df.apply(lambda row: 1 if row['築年数フラグ'] == 1 and row['家賃フラグ'] == 1 and row['アクセスフラグ'] == 1 else 0, axis=1)

    # 築年数フラグ、家賃フラグ、アクセスフラグのうちいずれか2つが1の行にサブターゲットフラグを1とする
    def sub_target_flag(row):
        count_true = sum([row['築年数フラグ'], row['家賃フラグ'], row['アクセスフラグ']])
        if count_true == 2:
            return 1
        else:
            return 0
    df.fillna(0, inplace=True)
    # 無限大の値を置き換える
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # ここで再度 NaN を取り除くか、他の有効な値に置き換える
    df.fillna(0, inplace=True)

    # アクセス列から駅名を抽出する関数
    def extract_station_name(access):
        match = re.search(r'/(.*?)駅', str(access))
        if match:
            return match.group(1)
        else:
            return None

    # アクセス2、アクセス3、アクセス4の各列から駅名を抽出し、新しい列 "駅名" に追加
    df['駅名'] = df['アクセス2'].apply(extract_station_name)
    df['駅名'] = df['駅名'].fillna(df['アクセス3'].apply(extract_station_name))
    df['駅名'] = df['駅名'].fillna(df['アクセス4'].apply(extract_station_name))

    df['サブターゲットフラグ'] = df.apply(sub_target_flag, axis=1)

    df.to_csv('output.csv', index=False, encoding='utf-8')
    st.text("スクレイピングが完了し、データをoutput.csvに保存しました。")


####フィルタリングコードを追加###################################################

    # output.csvからデータを読み込む
    #df = pd.read_csv('output.csv', encoding='utf-8')
    # # placeの値が存在する場合のみフィルタリングを行う
    # if place is not None and place != "":
    #     df = df[df['住所'].str.contains(place)]

    # # flagに応じてデータを抽出
    # if flag == "ターゲットフラグ":
    #     filtered_df = df[df['ターゲットフラグ'] == 1]
    #     if place is not None and place != "":
    #         filtered_df = filtered_df[filtered_df['住所'].str.contains(place)]
    # elif flag == "サブターゲットフラグ":
    #     filtered_df = df[df['サブターゲットフラグ'] == 1]
    #     if place is not None and place != "":
    #         filtered_df = filtered_df[filtered_df['住所'].str.contains(place)]
    # else:
    #     filtered_df = df  # フラグが選択されていない場合は全てのデータを表示

    # placeの値が存在する場合のみフィルタリングを行う


    # placeの値が存在する場合のみフィルタリングを行う
    if place is not None and place != "":
        df = df[df['住所'].str.contains(place)]
    
    # flagに応じてデータを抽出
    if flag == "ターゲットフラグ":
        filtered_df = df[df['ターゲットフラグ'] == 1]
    elif flag == "サブターゲットフラグ":
        filtered_df = df[df['サブターゲットフラグ'] == 1]
    else:
        filtered_df = df  # フラグが選択されていない場合は全てのデータを表示

    # yearに入力された築年数以下のデータを抽出
    if year is not None and year != "":
        year = int(year)  # ユーザー入力を整数に変換
        filterd_df = df[df['築年数_値'] <= year]

    # accessの値が空の場合はデータの絞り込みを行わない
    if access != "":
        # accessを整数に変換
        access_minutes = int(access)
        
        # min_minuteカラムの値がaccess_minutes以下のデータを抽出
        filtered_df = df[df['min_minute'] <= access_minutes]
    else:
        filtered_df = df  # accessが空の場合は全てのデータを表示

    # # 正規表現パターンを定義
    # pattern = r"東京都(.*区)"

    # # 正規表現でエリア名を抽出
    # match = re.search(pattern, place)

    # # エリア名が見つかった場合
    # if match:
    #     area_name = match.group(1)
    #     st.sidebar.text(f"抽出されたエリア: {area_name}")
        
    #     # フィルタリング条件に area_name を使用する
    #     df = df[df['住所'].str.contains(area_name)]
        
    #     # フィルタリングされたデータを表示
    #     st.dataframe(df)
    # else:
    #     st.sidebar.text("エリア名が見つかりませんでした。")


    # 抽出したデータをoutput_select.csvとして保存
    #Sfiltered_df = df  # フィルタリングしたデータをfiltered_dfに代入

    #filtered_df.to_csv('output_select.csv', index=False, encoding='utf-8')

    # 抽出したデータをoutput_select.csvとして保存
    df.to_csv('output_select.csv', index=False, encoding='utf-8')

    # ストリームリットに結果を表示
    st.text("データの抽出が完了し、output_select.csvに保存しました。")


    ##データ分析コード追加################################################
    # データ分析
    st.subheader("データ分析")

    ###################################################################
    #駅名×物件数のグラフを表示
    # 駅名ごとのカウントを取得
    #station_counts = data['駅名'].value_counts()
    # 駅名ごとの物件数を取得
    #station_counts = df['アクセス2'].apply(extract_station_name).value_counts()
    data = pd.read_csv('output_select.csv')  
    # 駅名ごとのカウントを取得
    station_counts = df['駅名'].value_counts()
    
    # グラフの作成
    plt.figure(figsize=(12, 6))
    plt.rcParams['font.family'] = 'MS Gothic'  # フォントをMS Gothicに設定=文字化け対策！

    ax = station_counts.plot(kind='bar')
    plt.title('駅名ごとの物件数', fontname='MS Gothic')
    plt.xlabel('駅名', fontname='MS Gothic')
    plt.ylabel('物件数', fontname='MS Gothic')
    plt.xticks(rotation=90)  # X軸のラベルを90度回転して読みやすくする

    # 凡例を日本語フォントで表示
    ax.legend(prop={'family': 'MS Gothic'})

    # グラフを表示
    st.pyplot(plt)

    ###################################################################
    #駅名×ターゲットフラグ合計数
    # 駅名ごとのカウントを取得
    # station_counts = df['駅名'].value_counts()

    # # 駅名を抽出する関数
    # def extract_station_name_from_access(access):
    #     match = re.search(r'(.*?)駅', str(access))
    #     if match:
    #         return match.group(1)
    #     else:
    #         return None

    # # 'アクセス2' 列から駅名だけを抽出
    # df['駅名'] = df['アクセス2'].apply(extract_station_name_from_access)

    # # 駅名ごとの「ターゲットフラグ」の合計数を取得
    # station_target_counts = df.groupby('駅名')['ターゲットフラグ'].sum()


    # # グラフの作成
    # plt.figure(figsize=(12, 6))
    # plt.rcParams['font.family'] = 'MS Gothic'  # フォントをMS Gothicに設定

    # station_target_counts.plot(kind='bar')
    # plt.title('駅名ごとのターゲットフラグ合計数', fontname='MS Gothic')
    # plt.xlabel('駅名', fontname='MS Gothic')
    # plt.ylabel('ターゲットフラグ合計数', fontname='MS Gothic')
    # plt.xticks(rotation=90)  # X軸のラベルを90度回転して読みやすくする

    # # グラフを表示
    # st.pyplot(plt)

    ###################################################################
    #駅名×サブターゲット数のグラフを表示

    # 駅名ごとのカウントを取得
    # station_counts = df['駅名'].value_counts()
    


    # # 駅名を抽出する関数
    # def extract_station_name_from_access(access):
    #     match = re.search(r'(.*?)駅', str(access))
    #     if match:
    #         return match.group(1)
    #     else:
    #         return None

    # # 'アクセス2' 列から駅名だけを抽出
    # df['駅名'] = df['アクセス2'].apply(extract_station_name_from_access)

    # # 駅名ごとの「サブターゲットフラグ」の合計数を取得
    # station_sub_target_counts = df.groupby('アクセス2')['サブターゲットフラグ'].sum()

    # # グラフの作成
    # plt.figure(figsize=(12, 6))
    # plt.rcParams['font.family'] = 'MS Gothic'  # フォントをMS Gothicに設定

    # station_sub_target_counts.plot(kind='bar')
    # plt.title('駅名ごとのサブターゲットフラグ合計数', fontname='MS Gothic')
    # plt.xlabel('駅名', fontname='MS Gothic')
    # plt.ylabel('サブターゲットフラグ合計数', fontname='MS Gothic')
    # plt.xticks(rotation=90)  # X軸のラベルを90度回転して読みやすくする

    # # グラフを表示
    # st.pyplot(plt)

    ###################################################################
    # データの読み込み
    data = pd.read_csv('output_select.csv')  
 
    # 各駅ごとの家賃/平米の平均値を棒グラフで表示
    
    # 各駅ごとの家賃/平米の平均値を計算
    station_avg_rent_per_sqm = data.groupby('駅名')['家賃/平米'].mean().reset_index()

    # Streamlitアプリケーションの開始
    st.title('各駅ごとの家賃/平米の平均値')

    # MatplotlibのFigureを作成
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams['font.family'] = 'MS Gothic'  # フォントをMS Gothicに設定

    # 各駅ごとの家賃/平米の平均値を棒グラフで表示
    ax.bar(station_avg_rent_per_sqm['駅名'], station_avg_rent_per_sqm['家賃/平米'], color='b', alpha=0.7)
    ax.set_xlabel('駅名', fontname='MS Gothic')
    ax.set_ylabel('家賃/平米', fontname='MS Gothic')
    ax.set_title('各駅ごとの家賃/平米の平均値', fontname='MS Gothic')
    plt.xticks(rotation=90)  # X軸のラベルを90度回転して読みやすくする

    # グラフを表示
    st.pyplot(fig)






    ####################################################################
    # データの読み込み
    data = pd.read_csv('output_select.csv')  

    # 各駅ごとの物件数を計算
    station_counts = data['駅名'].value_counts().reset_index()
    station_counts.columns = ['駅名', '物件数']

    # 各駅ごとの家賃/平米の平均値を計算
    station_avg_rent_per_sqm = data.groupby('駅名')['家賃/平米'].mean().reset_index()

    # 全体の家賃/平米の平均値を計算
    overall_avg_rent_per_sqm = data['家賃/平米'].mean()

    # Streamlitアプリケーションの開始
    st.title('各駅ごとの物件数と家賃/平米の関係')

    # MatplotlibのFigureを作成
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.rcParams['font.family'] = 'MS Gothic'  # フォントをMS Gothicに設定

    # 物件数の棒グラフを描画（左側の軸）
    ax1.bar(station_counts['駅名'], station_counts['物件数'], color='b', alpha=0.7)
    ax1.set_xlabel('駅名', fontname='MS Gothic')
    ax1.set_ylabel('物件数（件）', fontname='MS Gothic', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 家賃/平米の折れ線グラフを描画（右側の軸）
    ax2 = ax1.twinx()
    ax2.plot(station_avg_rent_per_sqm['駅名'], station_avg_rent_per_sqm['家賃/平米'], marker='o', color='r', linestyle='-', markersize=6)
    ax2.set_ylabel('家賃/平米（円）', fontname='MS Gothic', color='r')

    # 全体平均の家賃/平米の折れ線グラフを描画（右側の軸）
    ax2.axhline(overall_avg_rent_per_sqm, color='b', linestyle='--', label='全体平均', linewidth=2)

    # グラフタイトル
    plt.title('各駅ごとの物件数と家賃/平米の関係', fontname='MS Gothic')

    # 凡例を表示
    lines, labels = ax2.get_legend_handles_labels()
    ax2.legend(lines, labels, loc='upper left', fontsize='medium', title_fontsize='large', prop=FontProperties(fname='C:/Windows/Fonts/msyh.ttc'))

    # グラフを表示
    plt.xticks(rotation=90)  # X軸のラベルを90度回転して読みやすくする
    plt.tight_layout()

    # MatplotlibのFigureをStreamlitに表示
    st.pyplot(fig)

##################################################
#仮で地図を出してみる
    # df = pd.DataFrame(
    # np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    # columns=['lat', 'lon'])
    # st.map(df)

    # from geopy.geocoders import Nominatim

    # # ジオコーディングサービスを初期化
    # geolocator = Nominatim(user_agent="geoapiExercises")

    # # サンプルデータフレームを作成
    # data = {'住所': ['東京都千代田区', '大阪府大阪市', '京都府京都市']}
    # df = pd.DataFrame(data)

    # # 住所から緯度と経度を取得する関数を定義
    # def get_lat_lon(address):
    #     location = geolocator.geocode(address)
    #     if location:
    #         return (location.latitude, location.longitude)
    #     else:
    #         return None

    # # データフレームに新しいカラムを追加して緯度と経度を取得
    # df['緯度経度'] = df['住所'].apply(get_lat_lon)

    # # 地図上に位置をプロット
    # st.map(df)
