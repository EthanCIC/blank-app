import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# max FDV 時，no cliff and min vesting (1 month)，tge 全部釋放的 pv

def calculate_max_pv(min_FDV, discount_rate, max_cliff=12, max_vesting=24+12, tge_percentage=0.05):
    i = 1/(1+discount_rate/12)
    
    return min_FDV / (tge_percentage * i ** (max_cliff) + 
                     (1-tge_percentage) / (max_vesting-1) * 
                     i ** (max_cliff+1) / (1-i) * 
                     (1-i**(max_vesting-1)))

def calculate_pv(total_tokens, max_pv, discount_rate, cliff, vesting, tge):
    """
    計算每個月的現值 (PV) 和累積釋放量，並延伸到 36 個月
    """
    total_months = 36+12
    months = list(range(1, total_months + 1))
    
    tge_percentage = tge
    
    # 計算每月釋放金額
    # TGE 釋放量
    tge_reward = max_pv * tge_percentage
    # 剩餘金額平均分配到剩餘的 vesting 期間
    remaining_amount = max_pv * (1 - tge_percentage)
    monthly_reward = remaining_amount / (vesting - 1) if vesting > 1 else 0

    rewards = []
    pvs = []
    cumulative_tokens = []
    current_total = 0
    release_tokens = []
    
    for month in months:
        if month <= cliff:  # Cliff 期間
            reward_value = 0
            pv = 0
        elif month == cliff + 1:  # TGE 月（cliff 結束後的第一個月）
            reward_value = tge_percentage * max_pv
            pv = tge_reward / ((1 + discount_rate/12) ** (month - 1))
        elif month <= cliff + vesting:  # Vesting 期間
            reward_value = monthly_reward
            pv = reward_value / ((1 + discount_rate/12) ** (month - 1))
        else:  # 超過 vesting 期間
            reward_value = 0
            pv = 0
            
        current_total += reward_value
        rewards.append(reward_value)
        pvs.append(pv)
        cumulative_tokens.append(current_total)
        

    # 用 value 的比例來計算每個月實際的 token release
    release_tokens = [reward_value / sum(rewards) * total_tokens for reward_value in rewards]

    # 用 token released 計算累積 token released 
    cumulative_release_tokens = [sum(release_tokens[:i+1]) for i in range(len(release_tokens))]

    data = pd.DataFrame({
        "Month": months,
        "Value": rewards,
        "PV": pvs,
        "Cumulative Tokens": cumulative_tokens,
        "Token Released": release_tokens,
        "Cumulative Token Released": cumulative_release_tokens
    })
    return data

# Streamlit UI 介面
st.title("T-REX Publisher NFT Sale Calculator")

total_supply = 1000000000
base_emission = 0.2 * total_supply
total_nodes = 100000

# 在側邊欄放置輸入控制項
with st.sidebar:
    st.markdown("# Tier 1 NFT Sale")
    st.markdown("")
    st.markdown("Base Emission (Total): 20\% of Total Supply")
    min_FDV = st.number_input("Min DFV (when max cliff and vesting)", value=100, step=100)
    total_tokens = st.number_input("Base Reward for This Tier ($TREX)", value=40000000, step=10000000)
    total_token_percentage = total_tokens/(0.2*total_supply)
    st.markdown(f"佔比: {total_token_percentage*100:.0f}\% of Base Emission")
    st.markdown(f"Engagement Reward for This Tier ($TREX): {total_tokens * 15/20:,.0f}")

    node_amount = total_tokens/(0.2*total_supply)*total_nodes
    st.markdown(f"Node Amount: {node_amount:,.0f} out of 100,000")

    discount_rate = st.slider("Annually Discount Rate (92% = 0.92)", min_value=0.0, max_value=10.0, value=0.92, step=0.01)
    cliff = st.slider("Cliff Duration (months)", min_value=0, max_value=12, value=12)
    vesting = st.slider("Vesting Duration (months)", min_value=1, max_value=36, value=36)
    
    # 新增 linear vesting toggle
    is_linear = st.toggle("No additional initial unlock", value=False)
    
    # 當 vesting = 1 時，強制 TGE = 100% 並禁用滑桿
    if vesting == 1:
        tge_percentage = 100
        tge = 1.0
        st.slider("First-Month Unlock (%)", min_value=0, max_value=100, value=100, disabled=True)
    else:
        if is_linear:
            tge_percentage = (1/vesting) * 100
            st.slider("First-Month Unlock (%)", min_value=0, max_value=100, value=int(tge_percentage), disabled=True)
        else:
            tge_percentage = st.slider("First-Month Unlock (%)", min_value=0, max_value=100, value=5)
        tge = tge_percentage / 100

# 計算 PV
max_pv = calculate_max_pv(min_FDV, discount_rate, 12, 24+12, 1/vesting) 
# print(f"min_FDV: {min_FDV}")
# print(f"max_pv: {max_pv}")
result = calculate_pv(total_tokens, max_pv, discount_rate, cliff, vesting, tge)

# 計算 engagement reward 的累積釋放
engagement_tokens = total_tokens * 15/20
monthly_engagement = engagement_tokens / (36+12)
engagement_cumulative = [monthly_engagement * (i+1) for i in range(36+12)]

# 在主要區域顯示 Sale Price
sale_fdv = result['PV'].sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="FDV", value=f"${sale_fdv:,.0f}M")

with col2:
    node_price = sale_fdv*1e6*(total_tokens+engagement_tokens)/total_supply/node_amount
    st.metric(label="Node Price", value=f"${node_price:,.0f}")

with col3:
    st.metric(label="Node Amount", value=f"{node_amount:,.0f}")

with col4:
    st.metric(label="Income", value=f"${node_price*node_amount/1e6:,.2f}M")

# 繪製累積代幣釋放圖
st.subheader("累積代幣釋放圖")

# 建立新的 DataFrame 包含兩條線
plot_df = pd.DataFrame({
    "Month": result["Month"],
    "Base Reward": result["Cumulative Token Released"],
    "Engagement Reward": engagement_cumulative
})

# 使用 plotly 創建累積圖
fig = px.area(plot_df, 
    x="Month", 
    y=["Engagement Reward", "Base Reward"],
    title="累積代幣釋放圖",
    labels={"value": "Tokens"}
)

# 設定 x 軸為整數
fig.update_xaxes(
    dtick=1,
    tick0=0,
    tickmode='linear'
)

# 設定圖表樣式
fig.update_traces(line=dict(shape='hv'))

# 移除 legend 標題
fig.update_layout(
    legend_title_text="",  # 設定為空字串來移除 legend 標題
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# 顯示圖表
st.plotly_chart(fig, use_container_width=True)

# 在結果表格中加入 Engagement Reward
result['Monthly Engagement Reward'] = [monthly_engagement] * len(result)
result['Cumulative Engagement Reward'] = engagement_cumulative

# 顯示結果
st.subheader("計算結果明細")
st.dataframe(result)
