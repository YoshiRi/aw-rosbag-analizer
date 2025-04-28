import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========== 1. データ読み込み・前処理 関数 ===========
def get_xy_major_std(cov_6x6) -> float:
    """
    6x6 の共分散（1次元配列 [36]）から、
    (x,y) の2×2サブ行列を取り出して最大固有値の平方根を返す。

    - cov_6x6: shape=(36,) または (6,6) など。
      None や不正値の場合は None を返す。
    """
    # まず None や形状をチェック
    if cov_6x6 is None:
        return None
    arr = np.array(cov_6x6)
    if arr.size != 36:
        return None
    arr_2d = arr.reshape(6,6)

    # (x,y) 部分の2×2切り出し
    #   x軸 = arr_2d[0,0], arr_2d[0,1], arr_2d[1,0], arr_2d[1,1]
    sub_2x2 = arr_2d[:2, :2]  # 上左2×2
    # 固有値分解
    w, v = np.linalg.eig(sub_2x2)
    # 実数固有値の最大値をとる (負になってしまう可能性がある場合はabsを取るか要検討)
    max_eigenvalue = np.max(w)
    # eigenvalue が正であれば sqrt を返す
    if max_eigenvalue < 0:
        # 物理的にはありえない(負の分散)がデータエラーの場合は None など
        return None
    major_std = np.sqrt(max_eigenvalue)
    return major_std

def str2nparray(s:str):
    if isinstance(s, str):
        s = s.strip()
        # 先頭 '[' と末尾 ']' を除去（存在すれば）
        if s.startswith('['):
            s = s[1:]
        if s.endswith(']'):
            s = s[:-1]
        # 改行(\n)をスペースに置き換え
        s = s.replace('\n', ' ')
        # スペース区切りで float に変換
        arr = np.fromstring(s, sep=' ')
        return arr
    else:
        return None

@st.cache_data  # データをキャッシュすることで再読込みを防ぐ (Streamlit 1.18 以降は st.cache_data)
def load_data(csv_path: str) -> pd.DataFrame:
    """
    CSVを読み込み、最低限の前処理をして返す。
    """
    df = pd.read_csv(csv_path)
    
    # 例: timestampの最小値を0に合わせるなどの前処理
    t_min = df["timestamp"].min()
    df["t"] = (df["timestamp"] - t_min ) *1e-9  # ns -> s
    df["velocity"] = np.sqrt(df["velocity_x"]**2 + df["velocity_y"]**2)
    df["slip_angle"] = np.arctan2(df["velocity_y"], df["velocity_x"])
    df["distance"] = np.sqrt(df["position_x"]**2 + df["position_y"]**2)

    df["pose_covariance"] = df["pose_covariance"].apply(str2nparray)
    df["twist_covariance"] = df["twist_covariance"].apply(str2nparray)
    df["cov_std_x"] = df["pose_covariance"].apply(get_xy_major_std)
    df["cov_std_yaw"] = df["pose_covariance"].apply(lambda x: x[35] if x is not None else None)
    df["cov_std_v"] = df["twist_covariance"].apply(get_xy_major_std)
    df.drop(columns=["pose_covariance", "twist_covariance"], inplace=True)
    
    return df

# =========== 2. フィルタ用ウィジェット配置 関数 ===========

def sidebar_filters(df: pd.DataFrame):
    """
    サイドバーに各種フィルタ用のウィジェットを配置し、
    ユーザーが選択したフィルタ条件を返す。
    """

    st.sidebar.title("フィルタ設定")

    # ◆ Topicで絞り込み
    topic_list = sorted(df["topic"].dropna().unique())
    selected_topic = st.sidebar.selectbox("Select Topic", ["(All)"] + topic_list, index=1)

    # ◆ Labelで絞り込み
    label_list = sorted(df["label"].dropna().unique())
    selected_label = st.sidebar.selectbox("Select Label", ["(All)"] + label_list, index=0)

    # ◆ Object IDで絞り込み
    # object_ids = sorted(df["object_id"].dropna().unique())
    # selected_object_id = st.sidebar.selectbox("Select Object ID", ["(All)"] + object_ids, index=0)

    # ◆ timestampのスライダー設定
    t_min = df["t"].min()
    t_max = df["t"].max()
    
    # すべての時系列データを使用するオプション
    use_all_time_data = st.sidebar.checkbox("use all time series", value=False)
    
    # --- 2.1 中心時刻 (center_time) をスライダー or 数値入力 ---
    center_time = st.sidebar.slider(
        "Center Time [s]", 
        min_value=t_min, 
        max_value=t_max, 
        value= (t_min + t_max) / 2,  # 中心値
        disabled=use_all_time_data,  # すべてのデータを使用する場合は無効化
    )

    # --- 2.2 ウィンドウ長 (window_size) をスライダーや数値入力 ---
    # ここでは「全体の範囲を 0～(t_max - t_min)」に合わせてウィンドウを設定
    max_window_size = t_max - t_min
    window_size = st.sidebar.slider(
        "Window Size", 
        min_value=0.0, 
        max_value=20.0, 
        value=10.0,  # 1 sec
        step=0.1,
        disabled=use_all_time_data,  # すべてのデータを使用する場合は無効化
    )

    # ◆ x, y の範囲で絞り込み (スライダー)
    x_min, x_max = float(df["position_x"].min()), float(df["position_x"].max())
    selected_x_range = st.sidebar.slider(
        "X range",
        min_value=x_min,
        max_value=x_max,
        value=(x_min, x_max)
    )

    y_min, y_max = float(df["position_y"].min()), float(df["position_y"].max())
    selected_y_range = st.sidebar.slider(
        "Y range",
        min_value=y_min,
        max_value=y_max,
        value=(y_min, y_max)
    )

    # ◆ velocity の範囲で絞り込み (スライダー)
    v_min, v_max = float(df["velocity"].min()), float(df["velocity"].max())
    selected_v_range = st.sidebar.slider(
        "velocity range",
        min_value=v_min,
        max_value=v_max,
        value=(v_min, v_max)
    )

    # filter with distance
    d_min, d_max = float(df["distance"].min()), float(df["distance"].max())
    selected_d_range = st.sidebar.slider(
        "distance range",
        min_value=d_min,
        max_value=d_max,
        value=(d_min, d_max)
    )

    return {
        "topic": selected_topic,
        "label": selected_label,
        # "object_id": selected_object_id,
        "center_time": center_time,
        "window_size": window_size,
        "x_range": selected_x_range,
        "y_range": selected_y_range,
        "v_range": selected_v_range,
        "d_range": selected_d_range,
        "use_all_time_data": use_all_time_data,  # 新しいフラグを追加
    }

# =========== 3. フィルタロジックの適用 & 可視化 関数 ===========

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    受け取ったフィルタ条件を使って df を絞り込み、フィルタ後のデータを返す。
    """
    df_filtered = df.copy()

    # もし "(All)" 以外が選ばれていたら、絞り込み
    if filters["topic"] != "(All)":
        df_filtered = df_filtered[df_filtered["topic"] == filters["topic"]]

    if filters["label"] != "(All)":
        df_filtered = df_filtered[df_filtered["label"] == filters["label"]]

    # if filters["object_id"] != "(All)":
    #     df_filtered = df_filtered[df_filtered["object_id"] == filters["object_id"]]

    # すべての時系列データを使用するオプションがオフの場合のみ時間フィルタを適用
    if not filters["use_all_time_data"]:
        center_time = filters["center_time"]
        window_size = filters["window_size"]
        t_low = center_time - window_size/2
        t_high = center_time + window_size/2
        df_filtered = df_filtered[(df_filtered["t"] >= t_low) & (df_filtered["t"] <= t_high)]

    x_low, x_high = filters["x_range"]
    df_filtered = df_filtered[(df_filtered["position_x"] >= x_low) & (df_filtered["position_x"] <= x_high)]

    y_low, y_high = filters["y_range"]
    df_filtered = df_filtered[(df_filtered["position_y"] >= y_low) & (df_filtered["position_y"] <= y_high)]

    v_low, v_high = filters["v_range"]
    df_filtered = df_filtered[(df_filtered["velocity_x"] >= v_low) & (df_filtered["velocity_x"] <= v_high)]

    d_low, d_high = filters["d_range"]
    df_filtered = df_filtered[(df_filtered["distance"] >= d_low) & (df_filtered["distance"] <= d_high)]

    return df_filtered

def draw_object_id_slider(df: pd.DataFrame):
    """
    Object ID で絞り込むためのスライダーを描画する。
    """
    object_ids = df["object_id"].fillna("None").unique().tolist()
    selected_object_id = st.selectbox("Select Object ID", ["(All)"] + object_ids, index=0)
    if selected_object_id == "(All)":
        return df
    if selected_object_id == "None":
        df_filtered = df[df["object_id"].isnull()]
    else:
        df_filtered = df[df["object_id"] == selected_object_id]
    # for multi select
    # selected_obj_ids = st.multiselect(
    #     "Select object_ids from filtered data",
    #     options=object_ids,
    #     default=object_ids[0] if len(object_ids) > 0 else None
    # )
    # df_filtered = df[df["object_id"].isin(selected_obj_ids)]
    return df_filtered

def plot_scatter_position(df_filtered: pd.DataFrame, filters: dict):
    """
    フィルタ後のデータを受け取り、散布図をmatplotlibで作成して表示する。
    """
    fig, ax = plt.subplots(figsize=(12,8))

    if len(df_filtered) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
    else:
        ax.scatter(df_filtered["position_x"], df_filtered["position_y"], c='blue', alpha=0.5)
        
        # タイトル例: 選択中のフィルタ情報を少し入れる
        title_str = f"topic={filters['topic']}, label={filters['label']}"
        ax.set_title(title_str)
        
        ax.set_xlabel("position_x")
        ax.set_ylabel("position_y")

    st.pyplot(fig)

def plot_scatter_with_bbox(df_filtered, filters):
    """
    position_x, position_y を散布しつつ、
    length, width, yawを用いてバウンディングボックスを2D描画する。
    """
    fig, ax = plt.subplots(figsize=(12,8))

    if len(df_filtered) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        st.pyplot(fig)
        return

    # バウンディングボックス描画
    for _, row in df_filtered.iterrows():
        px = row["position_x"]
        py = row["position_y"]
        yaw = row["yaw"]
        length = row.get("length", None)
        width  = row.get("width", None)

        # length, width がない (NaN) 場合などのチェック
        if pd.isna(length) or pd.isna(width):
            # ここでは単純にスキップ
            continue

        # コーナーの頂点をローカル座標系で定義 (中心が(0,0)、前後左右に±length/2, ±width/2)
        corners = np.array([
            [ length/2,  width/2],
            [ length/2, -width/2],
            [-length/2, -width/2],
            [-length/2,  width/2],
            [ length/2,  width/2],  # 最終的に始点に戻って閉じる
        ])

        # yaw で回転させる回転行列
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation = np.array([
            [ cos_yaw, -sin_yaw ],
            [ sin_yaw,  cos_yaw ]
        ])

        # ローカルコーナー座標 -> 回転 -> グローバル座標
        corners_rotated = corners @ rotation.T
        corners_global = corners_rotated + np.array([px, py])

        # 辺を描画
        ax.plot(corners_global[:,0], corners_global[:,1], color='red', linewidth=1)

    title_str = f"topic={filters['topic']}, label={filters['label']}"
    ax.set_title(title_str)
    ax.set_xlabel("position_x")
    ax.set_ylabel("position_y")
    ax.set_aspect('equal', 'box')

    st.pyplot(fig)

def plot_time_series(df_filtered, filters):
    """
    横軸を時刻(t)として、縦軸に position_x, velocity_x, yaw などを重ね描画。
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    if len(df_filtered) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        st.pyplot(fig)
        return

    # ソートしておく (時刻順)
    df_filtered = df_filtered.sort_values("t")

    t = df_filtered["t"].values
    # ax.plot(t, df_filtered["position_x"].values, label="pos_x", color='blue', linestyle='-')
    # ax.plot(t, df_filtered["position_y"].values, label="pos_y", color='red', linestyle='-')
    ax.plot(t, df_filtered["distance"].values, label="dist", color='blue', linestyle='-', marker='.')

    ax.set_xlabel("time [relative]")
    ax.set_ylabel("value")
    ax.legend()
    title_str = f"Time series"
    ax.set_title(title_str)

    st.pyplot(fig)

    # velocity plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, df_filtered["velocity"].values, label="velocity", color='blue', linestyle='-', marker='.')
    ax.plot(t, df_filtered["cov_std_v"].values, label="cov_std_v", color='red', linestyle='-', marker='.')
    ax.plot(t, df_filtered["cov_std_x"].values, label="cov_std_x", color='green', linestyle='-', marker='.')
    ax.set_xlabel("time [relative]")
    ax.set_ylabel("velocity")
    ax.legend()
    title_str = f"Time series"
    ax.set_title(title_str)

    st.pyplot(fig)

    # yaw plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, df_filtered["yaw"].values, label="yaw", color='blue', linestyle='-', marker='.')
    ax.plot(t, df_filtered["cov_std_yaw"].values, label="cov_std_yaw", color='red', linestyle='-', marker='.')
    ax.plot(t, df_filtered["slip_angle"].values, label="slip_angle", color='green', linestyle='-', marker='.')
    ax.plot(t, df_filtered["yaw"].values + df_filtered["slip_angle"].values, label="velocity_direction", color='orange', linestyle='-', alpha=0.5)
    ax.set_xlabel("time [relative]")
    ax.set_ylabel("yaw")
    ax.legend()
    title_str = f"Time series"
    ax.set_title(title_str)

    st.pyplot(fig)

def plot_distance_histogram_by_class(df_filtered):
    """
    df_filtered の中で label (class) ごとに distance のヒストグラムを
    1枚のグラフに重ね描画する。
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if len(df_filtered) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        st.pyplot(fig)
        return

    # クラスごとにグループ化
    grouped = df_filtered.groupby("label")

    # bins は適宜変更
    bins = 30

    # 各クラスの distance を重ねて描画 (overlapped histogram)
    for label_value, sub_df in grouped:
        distances = sub_df["distance"].dropna()
        ax.hist(
            distances,
            bins=bins,
            alpha=0.5,            # 重ねたときに透けて見やすいように
            label=str(label_value)
        )

    ax.set_title("Distance Histogram by Class (Overlapped)")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

def plot_distance_histogram_by_class_stacked(df_filtered):
    """
    df_filtered の中で label (class) ごとに distance のヒストグラムを
    積み上げ（stacked）形式で1枚のグラフに描画する。
    """
    fig, ax = plt.subplots(figsize=(8,6))

    if len(df_filtered) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        st.pyplot(fig)
        return

    # クラスごとに distance データをリストで集める
    labels = df_filtered["label"].dropna().unique()
    distances_list = []

    # label順でリストを作成
    for label_value in labels:
        sub_df = df_filtered.loc[df_filtered["label"] == label_value, "distance"].dropna()
        distances_list.append(sub_df.values)

    # bins は適宜変更
    bins = 30

    # stacked=True で積み上げヒストグラム
    ax.hist(distances_list, bins=bins, stacked=True, label=labels)

    ax.set_title("Distance Histogram by Class (Stacked)")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.legend()
    
    st.pyplot(fig)

def plot_angle_histogram_by_class_stacked(df_filtered):
    """
    df_filtered の中で label (class) ごとに object の角度（arctan2(position_y, position_x)）
    のヒストグラムを積み上げ（stacked）形式で1枚のグラフに描画する。
    """
    fig, ax = plt.subplots(figsize=(8,6))
    
    if len(df_filtered) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        st.pyplot(fig)
        return

    # 各オブジェクトの角度を計算
    # np.arctan2を使うことで、正しく[-pi, pi]の範囲が得られる
    df_filtered = df_filtered.copy()  # オリジナルを変更しないためコピー
    df_filtered["angle"] = np.arctan2(df_filtered["position_y"], df_filtered["position_x"])

    # ラベルごとに角度データをリストで集める
    labels = df_filtered["label"].dropna().unique()
    angles_list = []
    for label_value in labels:
        sub_df = df_filtered.loc[df_filtered["label"] == label_value, "angle"].dropna()
        angles_list.append(sub_df.values)

    # ビンは 範囲によって[-pi, pi]か[-pi/2, pi/2] かを可変
    min_angle = df_filtered["angle"].min()
    max_angle = df_filtered["angle"].max()
    min_bin_angle = -np.pi if min_angle < -np.pi/2 else -np.pi/2
    max_bin_angle = np.pi if max_angle > np.pi/2 else np.pi/2
    # 30個のビンを基本として、範囲が広い場合は倍にする
    base_bin_num = 30
    bin_num = base_bin_num if max_bin_angle - min_bin_angle <= np.pi else base_bin_num * 2 
    bins = np.linspace(min_bin_angle, max_bin_angle, bin_num)
    
    # stacked=True で積み上げヒストグラム
    ax.hist(angles_list, bins=bins, stacked=True, label=labels)

    ax.set_title("Angle Histogram by Class (Stacked)")
    ax.set_xlabel("Angle (radian)")
    ax.set_ylabel("Count")
    ax.legend()
    
    st.pyplot(fig)

def plot_polar_distribution(df_filtered):
    """
    df_filtered の中で、各オブジェクトの角度 (arctan2(position_y, position_x))
    と distance の分布を極座標グリッド上にヒートマップ表示する。
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # コピーして安全に作業。既に 'distance' は前処理済みとする。
    df_plot = df_filtered.copy()
    
    # 角度がなければ計算する ([-π, π]の範囲)
    if "angle" not in df_plot.columns:
        df_plot["angle"] = np.arctan2(df_plot["position_y"], df_plot["position_x"])
    
    angles = df_plot["angle"].values
    distances = df_plot["distance"].values

    # 極座標のサブプロットを作成
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': 'polar'})

    # ビンの設定
    # 角度は[-π, π]の範囲を36ビン（10度刻み）に分割
    max_angle = df_plot["angle"].max()
    min_angle = df_plot["angle"].min()
    max_bin_angle = np.pi if max_angle > np.pi/2 else np.pi/2
    min_bin_angle = -np.pi if min_angle < -np.pi/2 else -np.pi/2
    bin_num = (max_bin_angle - min_bin_angle) / np.pi * 18
    theta_bins = np.linspace(min_bin_angle, max_bin_angle, int(bin_num)+1)
    # 距離は0から最大値まで20ビン（必要に応じて調整）
    max_distance = df_plot["distance"].max()
    max_bin_distance = 100 if max_distance < 110 else 200
    r_bins = np.linspace(0, max_bin_distance, 21)

    # 2次元ヒストグラムを作成。np.histogram2d の第一引数は角度、第二が距離
    H, theta_edges, r_edges = np.histogram2d(angles, distances, bins=[theta_bins, r_bins])
    # ※ H の形状は (len(theta_bins)-1, len(r_bins)-1) となる

    # pcolormesh に渡す際、H の軸の向きを合わせるために転置する
    pcm = ax.pcolormesh(theta_edges, r_edges, H.T, cmap='viridis')

    fig.colorbar(pcm, ax=ax, label='Count')
    ax.set_title("Polar Distribution of Angle and Distance")
    # 必要に応じて、ゼロの位置や方向を設定
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    st.pyplot(fig)

def plot_size_histogram_by_class(df_filtered):
    """
    df_filtered の中で label (class) ごとに物体のサイズ（length, width）の
    ヒストグラムを表示する。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if len(df_filtered) == 0:
        ax1.text(0.5, 0.5, "No data", ha='center', va='center')
        ax2.text(0.5, 0.5, "No data", ha='center', va='center')
        st.pyplot(fig)
        return
    
    # NaN値を除外
    df_size = df_filtered.dropna(subset=["length", "width"])
    
    if len(df_size) == 0:
        ax1.text(0.5, 0.5, "No size data", ha='center', va='center')
        ax2.text(0.5, 0.5, "No size data", ha='center', va='center')
        st.pyplot(fig)
        return
    
    # クラスごとにグループ化
    grouped = df_size.groupby("label")
    
    # 各クラスの length と width を描画
    for label_value, sub_df in grouped:
        lengths = sub_df["length"].values
        widths = sub_df["width"].values
        
        # length のヒストグラム
        ax1.hist(
            lengths,
            bins=20,
            alpha=0.5,
            label=str(label_value)
        )
        
        # width のヒストグラム
        ax2.hist(
            widths,
            bins=20,
            alpha=0.5,
            label=str(label_value)
        )
    
    ax1.set_title("Length Histogram by Class")
    ax1.set_xlabel("Length [m]")
    ax1.set_ylabel("Count")
    ax1.legend()
    
    ax2.set_title("Width Histogram by Class")
    ax2.set_xlabel("Width [m]")
    ax2.set_ylabel("Count")
    ax2.legend()
    
    st.pyplot(fig)

def plot_velocity_histogram_by_class(df_filtered):
    """
    df_filtered の中で label (class) ごとに物体の速度のヒストグラムを表示する。
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(df_filtered) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        st.pyplot(fig)
        return
    
    # クラスごとにグループ化
    grouped = df_filtered.groupby("label")
    
    # 各クラスの velocity を描画
    for label_value, sub_df in grouped:
        velocities = sub_df["velocity"].dropna().values
        
        if len(velocities) > 0:
            ax.hist(
                velocities,
                bins=25,
                alpha=0.5,
                label=str(label_value)
            )
    
    ax.set_title("Velocity Histogram by Class")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Count")
    ax.legend()
    
    st.pyplot(fig)


def main():
    st.title("Streamlit Filtering Example")

    # 1) データ読み込み・前処理
    df = load_data("output.csv")

    # 2) フィルタ用ウィジェットをサイドバーに表示・ユーザー入力受け取り
    filters = sidebar_filters(df)

    # 3) フィルタを適用して可視化
    df_filtered = apply_filters(df, filters)
    df_filtered = draw_object_id_slider(df_filtered)

    st.write(f"**Filtered data size**: {len(df_filtered)}")

    # 散布図の例
    # plot_scatter_position(df_filtered, filters)
    plot_scatter_with_bbox(df_filtered, filters)
    plot_time_series(df_filtered, filters)
    
    # 統計情報の表示
    st.header("統計情報")
    
    # タブを使って統計情報を整理
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["距離", "角度", "極座標分布", "サイズ", "速度"])
    
    with tab1:
        plot_distance_histogram_by_class_stacked(df_filtered)
    
    with tab2:
        plot_angle_histogram_by_class_stacked(df_filtered)
    
    with tab3:
        plot_polar_distribution(df_filtered)
    
    with tab4:
        plot_size_histogram_by_class(df_filtered)
    
    with tab5:
        plot_velocity_histogram_by_class(df_filtered)
if __name__ == "__main__":
    main()
