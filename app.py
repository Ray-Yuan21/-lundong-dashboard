#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业轮动策略 - 可视化Dashboard
提供数据更新、信号预测、可视化展示的完整界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import subprocess
import time

# ======================== 数据源配置 ========================
# 支持本地和 GitHub 两种数据源
DATA_SOURCE = os.getenv('DATA_SOURCE', 'github')  # 'local' 或 'github'
GITHUB_DATA_URL = os.getenv('GITHUB_DATA_URL', 
    'https://raw.githubusercontent.com/Ray-Yuan21/lundong-data/main')

print(f"[配置] 数据源: {DATA_SOURCE}")
if DATA_SOURCE == 'github':
    print(f"[配置] GitHub URL: {GITHUB_DATA_URL}")
# ==========================================================

# 添加路径 - 确保指向 lundong 目录
# 无论在哪里运行，都确保指向正确的项目根目录
if Path(__file__).name == 'app.py':
    # 从 rotation_dashboard/app.py 运行
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
else:
    # 其他情况，使用当前目录的父目录
    PROJECT_ROOT = Path.cwd().parent.resolve()

# 确保是 lundong 目录
if PROJECT_ROOT.name != 'lundong':
    # 如果不是，向上查找
    current = Path.cwd()
    while current.name != 'lundong' and current != current.parent:
        current = current.parent
    if current.name == 'lundong':
        PROJECT_ROOT = current
    else:
        # 最后的备选方案
        PROJECT_ROOT = Path('/home/ray/qlib/examples/lundong')

sys.path.append(str(PROJECT_ROOT))

# 设置页面配置
st.set_page_config(
    page_title="行业轮动策略Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


class RotationDashboard:
    """行业轮动Dashboard"""
    
    def __init__(self):
        self.data_dir = PROJECT_ROOT / "data"
        self.results_dir = PROJECT_ROOT / "relative_strength" / "factor_rotation" / "rotation_results"
        self.backtest_dir = PROJECT_ROOT / "relative_strength" / "factor_rotation" / "backtest_results"
        # 数据文件实际在 relative_strength 目录下
        self.processed_data_path = PROJECT_ROOT / "relative_strength" / "processed_industry_data.pkl"
        self.rotation_scores_path = self.results_dir / "rotation_scores.csv"
        self.selected_factors_path = self.results_dir / "selected_factors.csv"
        self.backtest_metrics_path = self.results_dir / "backtest_metrics.csv"
        # 增强回测结果路径
        self.period_returns_path = self.backtest_dir / "period_returns_top3_5d.csv"
        self.trade_signals_path = self.backtest_dir / "trade_signals_top3_5d.csv"
        self.enhanced_metrics_path = self.backtest_dir / "backtest_metrics_top3_5d.csv"
        
    def load_data_status(self):
        """加载数据状态"""
        if not self.processed_data_path.exists():
            return None, None
        
        df = pd.read_pickle(self.processed_data_path)
        
        # 处理多级索引的情况
        if isinstance(df.index, pd.MultiIndex):
            # 获取第一级索引（日期）的最大值
            latest_date = df.index.get_level_values(0).max()
        else:
            latest_date = df.index.max()
        
        # 确保是datetime类型
        if not isinstance(latest_date, pd.Timestamp):
            latest_date = pd.to_datetime(latest_date)
            
        data_count = len(df)
        
        return latest_date, data_count
    
    def load_rotation_scores(self):
        """加载轮动得分"""
        try:
            if DATA_SOURCE == 'github':
                url = f"{GITHUB_DATA_URL}/rotation_scores.csv"
                print(f"[加载] 从 GitHub 读取: {url}")
                df = pd.read_csv(url)
            else:
                if not self.rotation_scores_path.exists():
                    return None
                df = pd.read_csv(self.rotation_scores_path)
            
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"[错误] 加载轮动得分失败: {e}")
            return None
    
    def load_selected_factors(self):
        """加载选中的因子"""
        try:
            if DATA_SOURCE == 'github':
                url = f"{GITHUB_DATA_URL}/selected_factors.csv"
                print(f"[加载] 从 GitHub 读取: {url}")
                return pd.read_csv(url)
            else:
                if not self.selected_factors_path.exists():
                    return None
                return pd.read_csv(self.selected_factors_path)
        except Exception as e:
            print(f"[错误] 加载因子列表失败: {e}")
            return None
    
    def load_backtest_metrics(self):
        """加载回测指标"""
        try:
            if DATA_SOURCE == 'github':
                url = f"{GITHUB_DATA_URL}/backtest_results/backtest_metrics_top3_thu.csv"
                print(f"[加载] 从 GitHub 读取: {url}")
                return pd.read_csv(url, index_col=0)
            else:
                if not self.backtest_metrics_path.exists():
                    return None
                return pd.read_csv(self.backtest_metrics_path, index_col=0)
        except Exception as e:
            print(f"[提示] 回测指标文件不存在或加载失败: {e}")
            return None
    
    def load_period_returns(self):
        """加载周期收益率"""
        try:
            if DATA_SOURCE == 'github':
                url = f"{GITHUB_DATA_URL}/backtest_results/period_returns_top3_5d.csv"
                print(f"[加载] 从 GitHub 读取: {url}")
                df = pd.read_csv(url)
            else:
                if not self.period_returns_path.exists():
                    return None
                df = pd.read_csv(self.period_returns_path)
            
            if 'start_date' in df.columns:
                df['start_date'] = pd.to_datetime(df['start_date'])
            if 'end_date' in df.columns:
                df['end_date'] = pd.to_datetime(df['end_date'])
            return df
        except Exception as e:
            print(f"[提示] 周期收益率文件不存在: {e}")
            return None
    
    def load_trade_signals(self):
        """加载买卖信号"""
        try:
            if DATA_SOURCE == 'github':
                url = f"{GITHUB_DATA_URL}/backtest_results/trade_signals_top3_5d.csv"
                print(f"[加载] 从 GitHub 读取: {url}")
                df = pd.read_csv(url)
            else:
                if not self.trade_signals_path.exists():
                    return None
                df = pd.read_csv(self.trade_signals_path)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"[提示] 交易信号文件不存在: {e}")
            return None
    
    def load_enhanced_metrics(self):
        """加载增强回测指标"""
        try:
            if DATA_SOURCE == 'github':
                url = f"{GITHUB_DATA_URL}/backtest_results/backtest_metrics_top3_5d.csv"
                print(f"[加载] 从 GitHub 读取: {url}")
                return pd.read_csv(url)
            else:
                if not self.enhanced_metrics_path.exists():
                    return None
                return pd.read_csv(self.enhanced_metrics_path)
        except Exception as e:
            print(f"[提示] 增强回测指标文件不存在: {e}")
            return None
    
    def step1_download_data(self):
        """步骤1: 增量更新ETF数据（智能模式）"""
        import subprocess
        import sys
        import os
        
        try:
            python_exe = sys.executable
            project_root_abs = str(PROJECT_ROOT.resolve())
            
            data_dir = os.path.join(project_root_abs, "data")
            update_script = os.path.join(data_dir, "update.py")
            
            # 优先使用增量更新脚本
            result = subprocess.run(
                [python_exe, update_script],
                cwd=data_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return False, f"数据更新失败: {result.stderr[:200]}"
            
            return True, "数据更新成功！"
                
        except subprocess.TimeoutExpired:
            return False, "更新超时，请检查网络连接"
        except Exception as e:
            return False, f"更新异常: {str(e)}"
    
    def step2_preprocess_data(self):
        """步骤2: 预处理数据"""
        import subprocess
        import sys
        import os
        
        try:
            python_exe = sys.executable
            project_root_abs = str(PROJECT_ROOT.resolve())
            
            data_dir = os.path.join(project_root_abs, "data")
            preprocess_script = os.path.join(data_dir, "data_preprocessing.py")
            
            result = subprocess.run(
                [python_exe, preprocess_script],
                cwd=data_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return False, f"数据预处理失败: {result.stderr[:200]}"
            
            return True, "数据预处理完成！"
                
        except subprocess.TimeoutExpired:
            return False, "预处理超时"
        except Exception as e:
            return False, f"预处理异常: {str(e)}"
    
    def step3_factor_engineering(self):
        """步骤3: 计算技术因子"""
        import subprocess
        import sys
        import os
        
        try:
            python_exe = sys.executable
            project_root_abs = str(PROJECT_ROOT.resolve())
            
            factor_eng_dir = os.path.join(project_root_abs, "relative_strength", "factor_engineering")
            factor_eng_script = os.path.join(factor_eng_dir, "factor_engineering.py")
            
            result = subprocess.run(
                [python_exe, factor_eng_script],
                cwd=factor_eng_dir,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                return False, f"因子工程失败: {result.stderr[:200]}"
            
            return True, "技术因子计算完成！"
                
        except subprocess.TimeoutExpired:
            return False, "因子计算超时"
        except Exception as e:
            return False, f"因子计算异常: {str(e)}"
    
    def step4_factor_analysis(self):
        """步骤4: 分析因子有效性"""
        import subprocess
        import sys
        import os
        
        try:
            python_exe = sys.executable
            project_root_abs = str(PROJECT_ROOT.resolve())
            
            factor_eng_dir = os.path.join(project_root_abs, "relative_strength", "factor_engineering")
            factor_analysis_script = os.path.join(factor_eng_dir, "factor_analysis_fast.py")
            
            result = subprocess.run(
                [python_exe, factor_analysis_script],
                cwd=factor_eng_dir,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                return False, f"因子分析失败: {result.stderr[:200]}"
            
            return True, "因子分析完成！"
                
        except subprocess.TimeoutExpired:
            return False, "因子分析超时"
        except Exception as e:
            return False, f"因子分析异常: {str(e)}"
    
    def step5_generate_rotation_scores(self):
        """步骤5: 生成轮动得分"""
        import subprocess
        import sys
        import os
        
        try:
            python_exe = sys.executable
            project_root_abs = str(PROJECT_ROOT.resolve())
            
            rotation_dir = os.path.join(project_root_abs, "relative_strength", "factor_rotation")
            rotation_script = os.path.join(rotation_dir, "rotation_strategy.py")
            
            result = subprocess.run(
                [python_exe, rotation_script],
                cwd=rotation_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return False, f"生成轮动得分失败: {result.stderr[:200]}"
            
            return True, "轮动得分生成完成！"
                
        except subprocess.TimeoutExpired:
            return False, "轮动得分生成超时"
        except Exception as e:
            return False, f"轮动得分生成异常: {str(e)}"
    
    def step6_run_enhanced_backtest(self):
        """步骤6: 运行增强回测（计算周期收益和买卖点）"""
        import subprocess
        import sys
        import os
        
        try:
            python_exe = sys.executable
            project_root_abs = str(PROJECT_ROOT.resolve())
            
            rotation_dir = os.path.join(project_root_abs, "relative_strength", "factor_rotation")
            backtest_script = os.path.join(rotation_dir, "enhanced_backtest.py")
            
            result = subprocess.run(
                [python_exe, backtest_script, '--top_n', '3', '--rebalance_period', '5'],
                cwd=rotation_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return False, f"增强回测失败: {result.stderr[:200]}"
            
            return True, "增强回测完成！"
                
        except subprocess.TimeoutExpired:
            return False, "增强回测超时"
        except Exception as e:
            return False, f"增强回测异常: {str(e)}"
    
    def update_all(self):
        """一键完整更新（所有步骤）"""
        steps = [
            ("🔄 增量更新数据", self.step1_download_data, True),  # 允许失败
            ("🔧 预处理数据", self.step2_preprocess_data, False),
            ("⚙️ 计算因子", self.step3_factor_engineering, False),
            ("📈 分析因子", self.step4_factor_analysis, False),
            ("📊 生成信号", self.step5_generate_rotation_scores, False),
            ("🎯 运行回测", self.step6_run_enhanced_backtest, False),
        ]
        
        for i, (name, func, allow_fail) in enumerate(steps, 1):
            st.info(f"[{i}/6] {name}...")
            success, message = func()
            
            if success:
                st.success(f"✅ {message}")
            else:
                if allow_fail:
                    st.warning(f"⚠️ {message} (将使用现有数据)")
                else:
                    st.error(f"❌ {message}")
                    return False, f"步骤{i}失败: {message}"
        
        return True, "全部步骤完成！"


def main():
    """主函数"""
    
    # 显示路径信息（调试用）
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📁 项目路径: {PROJECT_ROOT}")
    
    # 初始化
    dashboard = RotationDashboard()
    
    # 标题
    st.markdown('<div class="main-header">📊 行业轮动策略 Dashboard</div>', unsafe_allow_html=True)
    
    # 侧边栏
    st.sidebar.title("⚙️ 控制面板")
    
    # 页面选择
    page = st.sidebar.radio(
        "选择功能",
        ["🏠 首页概览", "🔄 数据更新", "📈 信号预测", "📊 可视化分析", "⚙️ 策略配置"]
    )
    
    st.sidebar.markdown("---")
    
    # 数据状态
    st.sidebar.subheader("📊 数据状态")
    latest_date, data_count = dashboard.load_data_status()
    scores_df = dashboard.load_rotation_scores()
    
    if latest_date and scores_df is not None:
        # 数据存在
        st.sidebar.success(f"✅ 数据已加载")
        st.sidebar.info(f"📅 数据日期: {latest_date.strftime('%Y-%m-%d')}")
        
        # 信号日期
        signal_date = pd.to_datetime(scores_df['date'].max())
        st.sidebar.info(f"🎯 信号日期: {signal_date.strftime('%Y-%m-%d')}")
        
        # 数据年龄和建议
        days_old = (datetime.now() - latest_date).days
        if days_old <= 1:
            st.sidebar.success(f"🟢 数据很新 ({days_old}天)")
        elif days_old <= 3:
            st.sidebar.info(f"🟡 数据较新 ({days_old}天)")
        else:
            st.sidebar.warning(f"🔴 数据较旧 ({days_old}天)")
            st.sidebar.error("建议立即更新！")
        
        st.sidebar.caption(f"总计 {data_count:,} 条记录")
    else:
        st.sidebar.error("❌ 未找到数据")
        st.sidebar.warning("请先更新数据！")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("当前配置: 20日因子 + 5日调仓")
    
    # 根据选择显示不同页面
    if page == "🏠 首页概览":
        show_home_page(dashboard)
    elif page == "🔄 数据更新":
        show_update_page(dashboard)
    elif page == "📈 信号预测":
        show_signal_page(dashboard)
    elif page == "📊 可视化分析":
        show_analysis_page(dashboard)
    elif page == "⚙️ 策略配置":
        show_config_page(dashboard)


def show_home_page(dashboard):
    """首页概览"""
    
    st.header("🏠 策略概览")
    
    # 首先显示数据状态 - 这是最重要的！
    st.subheader("📊 当前数据状态")
    
    latest_date, data_count = dashboard.load_data_status()
    scores_df = dashboard.load_rotation_scores()
    
    if latest_date and scores_df is not None:
        # 数据存在，显示详细状态
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📅 数据日期", latest_date.strftime('%Y-%m-%d'))
        
        with col2:
            days_old = (datetime.now() - latest_date).days
            color = "normal" if days_old <= 2 else "inverse"
            st.metric("📈 数据年龄", f"{days_old} 天", delta_color=color)
        
        with col3:
            st.metric("📊 数据条数", f"{data_count:,}")
        
        with col4:
            signal_date = pd.to_datetime(scores_df['date'].max()).strftime('%Y-%m-%d')
            st.metric("🎯 信号日期", signal_date)
        
        # 数据状态提示
        if days_old <= 1:
            st.success("✅ 数据很新，可以直接使用")
        elif days_old <= 3:
            st.info("ℹ️ 数据较新，建议更新到最新")
        else:
            st.warning("⚠️ 数据较旧，强烈建议更新")
    else:
        # 数据不存在
        st.error("❌ 未找到数据文件！请先更新数据")
        st.info("💡 点击左侧「🔄 数据更新」→「🔄 立即更新数据」")
        return
    
    st.markdown("---")
    
    # 当前持仓推荐
    st.subheader("🎯 当前持仓推荐 (Top 3)")
    
    latest_scores = scores_df[scores_df['date'] == scores_df['date'].max()].copy()
    latest_scores = latest_scores.sort_values('rotation_score', ascending=False).head(3)
    
    # 显示推荐时间
    signal_time = pd.to_datetime(scores_df['date'].max())
    st.caption(f"基于 {signal_time.strftime('%Y年%m月%d日')} 的数据生成")
    
    for i, (idx, row) in enumerate(latest_scores.iterrows(), 1):
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"### {row['symbol']}")
        
        with col2:
            score = row['rotation_score']
            color = "green" if score > 0 else "red"
            st.markdown(f"<span style='color:{color}; font-size:1.5rem; font-weight:bold;'>{score:.4f}</span>", 
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"**排名 #{i}**")
    
    st.markdown("---")
    
    # 关键指标
    st.subheader("📈 策略表现")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("年化收益率", "37.71%", "+21.21%")
    
    with col2:
        st.metric("夏普比率", "1.29", "+0.57")
    
    with col3:
        st.metric("最大回撤", "-28.52%", "")
    
    with col4:
        st.metric("换手率", "41.6%", "")
    
    st.markdown("---")
    
    # 快速操作
    st.subheader("⚡ 快速操作")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if days_old > 2:
            st.error("🔄 建议更新数据")
        else:
            st.info("💡 点击左侧「🔄 数据更新」")
    
    with col2:
        st.success("📈 查看详细信号")
    
    with col3:
        st.info("📊 可视化分析")


def show_update_page(dashboard):
    """数据更新页面 - 分步骤执行"""
    
    st.header("🔄 数据更新中心")
    
    # 当前数据状态
    st.subheader("📊 当前数据状态")
    
    latest_date, data_count = dashboard.load_data_status()
    
    if latest_date:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("最新日期", latest_date.strftime('%Y-%m-%d'))
        
        with col2:
            st.metric("数据条数", f"{data_count:,}")
        
        with col3:
            days_old = (datetime.now() - latest_date).days
            st.metric("数据年龄", f"{days_old} 天")
    else:
        st.warning("⚠️ 未找到数据文件")
    
    st.markdown("---")
    
    # 更新方式选择
    update_mode = st.radio(
        "选择更新方式",
        ["🚀 一键完整更新", "🔧 分步执行（手动）"],
        help="完整更新会自动执行所有步骤；分步执行允许你选择性地执行某些步骤"
    )
    
    st.markdown("---")
    
    if update_mode == "🚀 一键完整更新":
        # 一键完整更新
        st.subheader("🚀 一键完整更新")
        
        st.info("""
        **完整更新流程 (6步):**
        1. 🔄 增量更新ETF数据 (智能) - ⏱️ ~30秒-1分钟
        2. 🔧 预处理数据 (清洗、格式化) - ⏱️ ~30秒
        3. ⚙️ 计算技术因子 (动量、反转、波动率等) - ⏱️ ~1分钟
        4. 📈 分析因子有效性 (IC、IR指标) - ⏱️ ~1分钟
        5. 📊 生成轮动得分 (选股信号) - ⏱️ ~30秒
        6. 🎯 运行增强回测 (周期收益、买卖点) - ⏱️ ~30秒
        
        💡 **提示**: 
        - 步骤1使用智能增量更新，只下载新数据，大幅降低IP被封风险
        - 如果数据已最新，会自动跳过下载
        - 步骤6会生成每个调仓周期的收益率和买卖点信号
        - 首次使用需在命令行运行: `python data/dowload.py`
        """)
        
        if st.button("🔄 开始完整更新", type="primary", use_container_width=True):
            with st.spinner("正在执行完整更新..."):
                success, message = dashboard.update_all()
                
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
    
    else:
        # 分步执行
        st.subheader("🔧 分步执行更新")
        
        st.info("""
        **适用场景**:
        - 📥 数据更新失败时，可以跳过步骤1，只更新因子
        - ⚡ 只想重新计算因子，不下载新数据
        - 🎯 调试某个特定步骤
        - 🔍 查看每个步骤的详细执行情况
        
        **执行顺序建议**: 步骤1 → 步骤2 → 步骤3 → 步骤4 → 步骤5
        
        **💡 关于步骤1**:
        - 使用智能增量更新，只下载缺失的新数据
        - 如果数据已是最新，会自动跳过
        - 首次使用需先运行全量下载: `cd data && python dowload.py`
        """)
        
        st.markdown("---")
        
        # 步骤1: 增量更新数据
        with st.expander("📥 **步骤1: 增量更新ETF数据**", expanded=True):
            st.caption("⏱️ 预计耗时: 30秒-1分钟 | 可跳过（使用现有数据）")
            st.markdown("""
            - 🔄 **智能增量更新**：只下载新数据，降低IP被封风险
            - 📊 数据源: AkShare (东方财富)
            - ✅ 如果数据已最新，自动跳过
            - 💡 首次使用需先运行全量下载（命令行: `python dowload.py`）
            """)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("▶️ 执行步骤1", key="step1", use_container_width=True):
                    with st.spinner("正在下载数据..."):
                        success, message = dashboard.step1_download_data()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                            st.info("💡 提示: 如果下载失败，可以跳过此步骤，使用现有数据继续")
            with col2:
                st.caption("状态:")
                if (dashboard.data_dir / "industry_index_data.pkl").exists():
                    st.success("✓ 已有数据")
                else:
                    st.warning("✗ 无数据")
        
        # 步骤2: 预处理
        with st.expander("🔧 **步骤2: 预处理数据**"):
            st.caption("⏱️ 预计耗时: 30秒 | 必需步骤")
            st.markdown("""
            - 数据清洗和格式化
            - 处理缺失值
            - 生成标准化数据
            """)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("▶️ 执行步骤2", key="step2", use_container_width=True):
                    with st.spinner("正在预处理数据..."):
                        success, message = dashboard.step2_preprocess_data()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
            with col2:
                st.caption("状态:")
                if dashboard.processed_data_path.exists():
                    st.success("✓ 已完成")
                else:
                    st.warning("✗ 未完成")
        
        # 步骤3: 计算因子
        with st.expander("⚙️ **步骤3: 计算技术因子**"):
            st.caption("⏱️ 预计耗时: 1分钟 | 必需步骤")
            st.markdown("""
            - 动量因子 (5日、20日、60日收益率)
            - 反转因子 (短期反转指标)
            - 波动率因子 (历史波动率)
            - 成交量因子 (量价配合指标)
            """)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("▶️ 执行步骤3", key="step3", use_container_width=True):
                    with st.spinner("正在计算因子..."):
                        success, message = dashboard.step3_factor_engineering()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
            with col2:
                st.caption("状态:")
                factor_file = PROJECT_ROOT / "relative_strength" / "factor_engineering" / "factor_data.pkl"
                if factor_file.exists():
                    st.success("✓ 已完成")
                else:
                    st.warning("✗ 未完成")
        
        # 步骤4: 因子分析
        with st.expander("📈 **步骤4: 分析因子有效性**"):
            st.caption("⏱️ 预计耗时: 1分钟 | 必需步骤")
            st.markdown("""
            - 计算因子IC (信息系数)
            - 计算因子IR (信息比率)
            - 筛选有效因子
            """)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("▶️ 执行步骤4", key="step4", use_container_width=True):
                    with st.spinner("正在分析因子..."):
                        success, message = dashboard.step4_factor_analysis()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
            with col2:
                st.caption("状态:")
                analysis_file = PROJECT_ROOT / "relative_strength" / "factor_engineering" / "factor_analysis_results.pkl"
                if analysis_file.exists():
                    st.success("✓ 已完成")
                else:
                    st.warning("✗ 未完成")
        
        # 步骤5: 生成轮动得分
        with st.expander("📊 **步骤5: 生成轮动得分**"):
            st.caption("⏱️ 预计耗时: 30秒 | 必需步骤")
            st.markdown("""
            - 基于有效因子生成综合得分
            - 生成行业排名
            - 输出选股信号
            """)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("▶️ 执行步骤5", key="step5", use_container_width=True):
                    with st.spinner("正在生成轮动得分..."):
                        success, message = dashboard.step5_generate_rotation_scores()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
            with col2:
                st.caption("状态:")
                if dashboard.rotation_scores_path.exists():
                    st.success("✓ 已完成")
                else:
                    st.warning("✗ 未完成")
        
        # 步骤6: 运行增强回测
        with st.expander("🎯 **步骤6: 运行增强回测**"):
            st.caption("⏱️ 预计耗时: 30秒 | 必需步骤")
            st.markdown("""
            - 计算每个调仓周期的收益率
            - 生成买卖点信号
            - 实时更新回测结果
            - 支持周期收益可视化
            """)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("▶️ 执行步骤6", key="step6", use_container_width=True):
                    with st.spinner("正在运行增强回测..."):
                        success, message = dashboard.step6_run_enhanced_backtest()
                        if success:
                            st.success(f"✅ {message}")
                            st.balloons()
                            st.info("🎉 所有步骤完成！可以查看最新的回测结果和买卖点了")
                        else:
                            st.error(f"❌ {message}")
            with col2:
                st.caption("状态:")
                if dashboard.period_returns_path.exists():
                    st.success("✓ 已完成")
                else:
                    st.warning("✗ 未完成")
        
        st.markdown("---")
        
        # 快速操作
        st.subheader("⚡ 快速操作")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 只更新因子（跳过下载）", use_container_width=True):
                """跳过数据下载，只更新因子和信号"""
                with st.spinner("正在执行步骤2-5..."):
                    steps = [
                        ("预处理数据", dashboard.step2_preprocess_data),
                        ("计算因子", dashboard.step3_factor_engineering),
                        ("分析因子", dashboard.step4_factor_analysis),
                        ("生成信号", dashboard.step5_generate_rotation_scores),
                    ]
                    
                    all_success = True
                    for name, func in steps:
                        st.info(f"执行: {name}...")
                        success, message = func()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                            all_success = False
                            break
                    
                    if all_success:
                        st.success("✅ 因子更新完成！")
                        st.balloons()
        
        with col2:
            if st.button("📊 只生成信号（跳过因子计算）", use_container_width=True):
                """只重新生成轮动得分"""
                with st.spinner("正在生成轮动得分..."):
                    success, message = dashboard.step5_generate_rotation_scores()
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")
    
    st.markdown("---")
    
    # 显示更新后的状态
    st.subheader("📋 当前数据状态")
    
    latest_date_new, data_count_new = dashboard.load_data_status()
    
    if latest_date_new:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最新日期", latest_date_new.strftime('%Y-%m-%d'))
        with col2:
            st.metric("数据条数", f"{data_count_new:,}")
        with col3:
            days_old = (datetime.now() - latest_date_new).days
            st.metric("数据年龄", f"{days_old} 天")
    else:
        st.warning("未找到数据文件")
    
    # 输出文件状态
    st.markdown("---")
    st.subheader("📁 输出文件状态")
    
    # 检查各个输出文件
    files_to_check = [
        ("原始数据", dashboard.processed_data_path, "relative_strength/processed_industry_data.pkl"),
        ("技术因子", PROJECT_ROOT / "layered_factors_v2.pkl", "layered_factors_v2.pkl"),
        ("因子分析", PROJECT_ROOT / "relative_strength/factor_engineering/factor_analysis_results/factor_performance.csv", "factor_performance.csv"),
        ("轮动得分", dashboard.rotation_scores_path, "rotation_scores.csv"),
        ("选中因子", dashboard.selected_factors_path, "selected_factors.csv"),
        ("回测指标", dashboard.backtest_metrics_path, "backtest_metrics.csv")
    ]
    
    col1, col2, col3 = st.columns(3)
    
    for i, (name, path, filename) in enumerate(files_to_check):
        with [col1, col2, col3][i % 3]:
            if path.exists():
                # 获取文件修改时间
                import os
                mtime = os.path.getmtime(path)
                mod_date = datetime.fromtimestamp(mtime)
                days_old = (datetime.now() - mod_date).days
                
                if days_old == 0:
                    status_color = "🟢"
                    status_text = "今天"
                elif days_old <= 1:
                    status_color = "🟡"
                    status_text = f"{days_old}天前"
                else:
                    status_color = "🔴"
                    status_text = f"{days_old}天前"
                
                st.metric(
                    label=f"{status_color} {name}",
                    value=status_text,
                    delta=filename
                )
            else:
                st.metric(
                    label=f"❌ {name}",
                    value="不存在",
                    delta=filename
                )
    
    # 历史更新记录
    st.markdown("---")
    st.subheader("📝 更新历史")
    
    if dashboard.rotation_scores_path.exists():
        scores_df = dashboard.load_rotation_scores()
        update_dates = scores_df['date'].unique()
        
        st.info(f"总共有 {len(update_dates)} 个交易日的数据")
        
        # 显示最近10次更新
        recent_dates = sorted(update_dates, reverse=True)[:10]
        
        df_display = pd.DataFrame({
            '日期': [pd.to_datetime(d).strftime('%Y-%m-%d') for d in recent_dates],
            '状态': ['✅ 已完成'] * len(recent_dates)
        })
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)


def show_signal_page(dashboard):
    """信号预测页面"""
    
    st.header("📈 轮动信号")
    
    scores_df = dashboard.load_rotation_scores()
    
    if scores_df is None:
        st.warning("⚠️ 未找到轮动得分数据，请先更新数据")
        return
    
    # 日期选择
    available_dates = sorted(scores_df['date'].unique(), reverse=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_date = st.selectbox(
            "选择日期",
            available_dates,
            format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')
        )
    
    with col2:
        top_n = st.slider("显示Top N", 3, 20, 10)
    
    # 获取当天得分
    daily_scores = scores_df[scores_df['date'] == selected_date].copy()
    daily_scores = daily_scores.sort_values('rotation_score', ascending=False)
    
    # Top N 推荐
    st.subheader(f"🎯 Top {top_n} 行业推荐")
    
    top_scores = daily_scores.head(top_n)
    
    # 柱状图
    fig = go.Figure()
    
    colors = ['#2ecc71' if score > 0 else '#e74c3c' 
              for score in top_scores['rotation_score']]
    
    fig.add_trace(go.Bar(
        x=top_scores['symbol'],
        y=top_scores['rotation_score'],
        marker_color=colors,
        text=top_scores['rotation_score'].round(4),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"轮动得分 Top {top_n} ({pd.to_datetime(selected_date).strftime('%Y-%m-%d')})",
        xaxis_title="行业",
        yaxis_title="轮动得分",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 详细表格
    st.subheader("📋 详细得分")
    
    display_df = daily_scores[['symbol', 'rotation_score']].copy()
    display_df['排名'] = range(1, len(display_df) + 1)
    display_df = display_df[['排名', 'symbol', 'rotation_score']]
    display_df.columns = ['排名', '行业', '轮动得分']
    
    st.dataframe(
        display_df.head(top_n),
        use_container_width=True,
        hide_index=True
    )
    
    # 下载按钮
    csv = daily_scores.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "📥 下载完整得分",
        csv,
        f"rotation_scores_{pd.to_datetime(selected_date).strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=True
    )


def show_analysis_page(dashboard):
    """可视化分析页面"""
    
    st.header("📊 可视化分析")
    
    scores_df = dashboard.load_rotation_scores()
    
    if scores_df is None:
        st.warning("⚠️ 未找到数据，请先更新")
        return
    
    # 选项卡
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 得分趋势", 
        "🏆 行业排名", 
        "📊 因子分析",
        "💰 周期收益率",
        "🎯 买卖点分析"
    ])
    
    with tab1:
        show_score_trend(scores_df)
    
    with tab2:
        show_industry_ranking(scores_df)
    
    with tab3:
        show_factor_analysis(dashboard)
    
    with tab4:
        show_period_returns(dashboard)
    
    with tab5:
        show_trade_signals(dashboard)


def show_score_trend(scores_df):
    """显示得分趋势"""
    
    st.subheader("📈 行业轮动得分趋势")
    
    # 行业选择
    industries = sorted(scores_df['symbol'].unique())
    
    selected_industries = st.multiselect(
        "选择行业（最多5个）",
        industries,
        default=industries[:3] if len(industries) >= 3 else industries
    )
    
    if not selected_industries:
        st.warning("请至少选择一个行业")
        return
    
    # 绘制趋势图
    fig = go.Figure()
    
    for industry in selected_industries[:5]:
        industry_data = scores_df[scores_df['symbol'] == industry].sort_values('date')
        
        fig.add_trace(go.Scatter(
            x=industry_data['date'],
            y=industry_data['rotation_score'],
            mode='lines+markers',
            name=industry,
            line=dict(width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="行业轮动得分时序图",
        xaxis_title="日期",
        yaxis_title="轮动得分",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_industry_ranking(scores_df):
    """显示行业排名变化"""
    
    st.subheader("🏆 行业排名热力图")
    
    # 计算每日排名
    ranking_data = []
    
    for date in sorted(scores_df['date'].unique()):
        daily = scores_df[scores_df['date'] == date].sort_values('rotation_score', ascending=False)
        daily['rank'] = range(1, len(daily) + 1)
        ranking_data.append(daily[['date', 'symbol', 'rank']])
    
    ranking_df = pd.concat(ranking_data)
    
    # 透视表
    pivot_df = ranking_df.pivot(index='symbol', columns='date', values='rank')
    
    # 只显示最近30天
    recent_cols = sorted(pivot_df.columns, reverse=True)[:30]
    pivot_df = pivot_df[sorted(recent_cols)]
    
    # 热力图
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=[pd.to_datetime(d).strftime('%m-%d') for d in pivot_df.columns],
        y=pivot_df.index,
        colorscale='RdYlGn_r',
        text=pivot_df.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="排名")
    ))
    
    fig.update_layout(
        title="行业排名变化（最近30天）",
        xaxis_title="日期",
        yaxis_title="行业",
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("颜色越绿代表排名越靠前，越红代表排名越靠后")


def show_factor_analysis(dashboard):
    """显示因子分析"""
    
    st.subheader("📊 选中因子分析")
    
    factors_df = dashboard.load_selected_factors()
    
    if factors_df is None:
        st.warning("⚠️ 未找到因子数据")
        return
    
    # 显示因子信息
    st.dataframe(factors_df, use_container_width=True, hide_index=True)
    
    # 因子权重可视化
    if 'weight' in factors_df.columns:
        fig = go.Figure(data=[
            go.Bar(
                x=factors_df['factor'],
                y=factors_df['weight'],
                marker_color='#3498db',
                text=factors_df['weight'].round(4),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="因子权重分布",
            xaxis_title="因子",
            yaxis_title="权重",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_config_page(dashboard):
    """策略配置页面"""
    
    st.header("⚙️ 策略配置")
    
    st.info("当前版本暂不支持动态修改配置，请直接修改配置文件")
    
    # 显示当前配置
    st.subheader("📋 当前配置")
    
    config_data = {
        "参数": ["因子周期", "调仓周期", "持仓数量", "因子选择数量", "最小IC_IR", "最小胜率"],
        "数值": ["20日", "5日", "3", "10", "0.05", "0.52"]
    }
    
    st.table(pd.DataFrame(config_data))
    
    st.markdown("---")
    
    # 配置文件路径
    st.subheader("📝 配置文件")
    
    config_file = PROJECT_ROOT / "relative_strength" / "factor_rotation" / "rotation_strategy.py"
    
    st.code(f"配置文件路径: {config_file}")
    
    st.markdown("""
    **修改方法:**
    1. 打开 `rotation_strategy.py`
    2. 修改 `main()` 函数中的参数
    3. 重新运行 `./daily_update.sh`
    """)


def show_period_returns(dashboard):
    """显示周期收益率"""
    
    st.subheader("💰 调仓周期收益率分析")
    
    period_df = dashboard.load_period_returns()
    
    if period_df is None:
        st.warning("⚠️ 未找到周期收益率数据，请先运行增强回测 (步骤6)")
        st.info("💡 前往「🔄 数据更新」→「步骤6: 运行增强回测」")
        return
    
    # 显示统计信息
    st.markdown("### 📊 周期统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总调仓次数", f"{len(period_df)}")
    
    with col2:
        avg_return = period_df['period_return'].mean()
        st.metric("平均周期收益", f"{avg_return:.2%}")
    
    with col3:
        win_rate = (period_df['period_return'] > 0).sum() / len(period_df)
        st.metric("周期胜率", f"{win_rate:.2%}")
    
    with col4:
        max_return = period_df['period_return'].max()
        st.metric("最大周期收益", f"{max_return:.2%}")
    
    st.markdown("---")
    
    # 周期收益率柱状图
    st.markdown("### 📊 每个周期的收益率")
    
    fig = go.Figure()
    
    # 添加柱状图
    colors = ['green' if r > 0 else 'red' for r in period_df['period_return']]
    
    fig.add_trace(go.Bar(
        x=period_df['period_number'],
        y=period_df['period_return'] * 100,  # 转换为百分比
        marker_color=colors,
        text=[f"{r:.2f}%" for r in period_df['period_return'] * 100],
        textposition='outside',
        hovertemplate='<b>周期 #%{x}</b><br>' +
                      '收益率: %{y:.2f}%<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title="每个调仓周期的收益率 (5日)",
        xaxis_title="周期编号",
        yaxis_title="收益率 (%)",
        height=500,
        showlegend=False,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 累积收益曲线
    st.markdown("### 📈 累积收益曲线")
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=period_df['period_number'],
        y=period_df['cumulative_value'],
        mode='lines+markers',
        name='累积净值',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6),
        hovertemplate='<b>周期 #%{x}</b><br>' +
                      '累积净值: %{y:.4f}<br>' +
                      '<extra></extra>'
    ))
    
    fig2.update_layout(
        title="累积净值曲线",
        xaxis_title="周期编号",
        yaxis_title="累积净值",
        height=400,
        showlegend=False,
        hovermode='closest'
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # 详细表格
    st.markdown("### 📋 周期详情")
    
    # 显示选项
    show_all = st.checkbox("显示全部周期", value=False)
    
    display_df = period_df.copy()
    
    # 处理列表列显示
    if 'positions' in display_df.columns:
        display_df['持仓'] = display_df['positions'].apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) else ', '.join(x))
    
    # 选择要显示的列
    display_df = display_df[[
        'period_number', 'start_date', 'end_date', 
        'period_return', 'cumulative_value', '持仓'
    ]].copy()
    
    display_df.columns = ['周期', '开始日期', '结束日期', '周期收益率', '累积净值', '持仓']
    
    # 格式化
    display_df['周期收益率'] = display_df['周期收益率'].apply(lambda x: f"{x:.2%}")
    display_df['累积净值'] = display_df['累积净值'].apply(lambda x: f"{x:.4f}")
    display_df['开始日期'] = pd.to_datetime(display_df['开始日期']).dt.strftime('%Y-%m-%d')
    display_df['结束日期'] = pd.to_datetime(display_df['结束日期']).dt.strftime('%Y-%m-%d')
    
    if not show_all:
        # 只显示最近20个周期
        display_df = display_df.tail(20)
        st.caption("显示最近20个周期")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # 下载按钮
    csv = period_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "📥 下载完整周期数据",
        csv,
        "period_returns.csv",
        "text/csv",
        use_container_width=True
    )


def show_trade_signals(dashboard):
    """显示买卖点分析"""
    
    st.subheader("🎯 买卖点信号分析")
    
    signals_df = dashboard.load_trade_signals()
    
    if signals_df is None:
        st.warning("⚠️ 未找到买卖信号数据，请先运行增强回测 (步骤6)")
        st.info("💡 前往「🔄 数据更新」→「步骤6: 运行增强回测」")
        return
    
    # 统计信息
    st.markdown("### 📊 信号统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_signals = len(signals_df)
        st.metric("总信号数", f"{total_signals}")
    
    with col2:
        buy_signals = (signals_df['action'] == 'BUY').sum()
        st.metric("买入信号", f"{buy_signals}", delta="🟢")
    
    with col3:
        sell_signals = (signals_df['action'] == 'SELL').sum()
        st.metric("卖出信号", f"{sell_signals}", delta="🔴")
    
    with col4:
        unique_symbols = signals_df['symbol'].nunique()
        st.metric("涉及行业", f"{unique_symbols}")
    
    st.markdown("---")
    
    # 交互式买卖点图表
    st.markdown("### 📈 买卖点可视化")
    
    # 选择行业
    all_symbols = sorted(signals_df['symbol'].unique())
    selected_symbol = st.selectbox(
        "选择行业查看买卖点",
        all_symbols,
        help="选择一个行业查看其历史买卖点"
    )
    
    if selected_symbol:
        # 筛选该行业的信号
        symbol_signals = signals_df[signals_df['symbol'] == selected_symbol].copy()
        
        if len(symbol_signals) == 0:
            st.warning(f"未找到 {selected_symbol} 的买卖信号")
            return
        
        st.success(f"✓ 找到 {len(symbol_signals)} 个信号记录")
        
        # 显示信号表格
        st.markdown("#### 📋 买卖信号记录")
        st.dataframe(
            symbol_signals.sort_values('date', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # GitHub 模式下只显示表格，本地模式才显示价格图
        if DATA_SOURCE == 'github':
            st.info("💡 提示：在线版本仅显示买卖信号表格。如需查看价格曲线图，请在本地运行 Dashboard。")
            return
        
        # 尝试加载价格数据（仅本地模式）
        try:
            # 尝试多个可能的路径
            possible_paths = [
                PROJECT_ROOT / "layered_factors_v2.pkl",
                PROJECT_ROOT / "relative_strength" / "layered_factors_v2.pkl"
            ]
            
            factor_data = None
            factor_data_path = None
            
            for path in possible_paths:
                if path.exists():
                    factor_data_path = path
                    factor_data = pd.read_pickle(path)
                    break
            
            if factor_data is None:
                st.error("❌ 未找到因子数据文件")
                st.info("💡 请确保已运行完整的数据更新流程")
                st.code(f"尝试的路径:\n" + "\n".join([str(p) for p in possible_paths]))
                return
            
            # 提取该行业的价格数据
            available_symbols = factor_data.index.get_level_values('symbol').unique().tolist()
            
            if selected_symbol not in available_symbols:
                st.warning(f"❌ 在因子数据中未找到 {selected_symbol}")
                st.info(f"💡 可用的行业 ({len(available_symbols)}个):")
                st.write(available_symbols)
                return
            
            # 提取价格数据
            price_data = factor_data.loc[(slice(None), selected_symbol), 'close'].reset_index()
            price_data.columns = ['date', 'symbol', 'close']
            price_data['date'] = pd.to_datetime(price_data['date'])
            
            # 绘制价格曲线和买卖点
            fig = go.Figure()
            
            # 价格曲线
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=price_data['close'],
                mode='lines',
                name='价格',
                line=dict(color='steelblue', width=2),
                hovertemplate='日期: %{x|%Y-%m-%d}<br>价格: %{y:.2f}<extra></extra>'
            ))
            
            # 买入点
            buy_signals = symbol_signals[symbol_signals['action'] == 'BUY'].copy()
            if len(buy_signals) > 0:
                buy_prices = []
                buy_dates = []
                
                for _, signal in buy_signals.iterrows():
                    signal_date = pd.to_datetime(signal['date'])
                    price_match = price_data[price_data['date'] == signal_date]
                    
                    if len(price_match) > 0:
                        buy_prices.append(price_match.iloc[0]['close'])
                        buy_dates.append(signal_date)
                    else:
                        # 找最近的价格
                        nearest = price_data[price_data['date'] >= signal_date].head(1)
                        if len(nearest) > 0:
                            buy_prices.append(nearest.iloc[0]['close'])
                            buy_dates.append(signal_date)
                
                if len(buy_prices) > 0:
                    fig.add_trace(go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode='markers',
                        name='买入',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='green',
                            line=dict(color='darkgreen', width=2)
                        ),
                        hovertemplate='<b>买入</b><br>日期: %{x|%Y-%m-%d}<br>价格: %{y:.2f}<extra></extra>'
                    ))
            
            # 卖出点
            sell_signals = symbol_signals[symbol_signals['action'] == 'SELL'].copy()
            if len(sell_signals) > 0:
                sell_prices = []
                sell_dates = []
                
                for _, signal in sell_signals.iterrows():
                    signal_date = pd.to_datetime(signal['date'])
                    price_match = price_data[price_data['date'] == signal_date]
                    
                    if len(price_match) > 0:
                        sell_prices.append(price_match.iloc[0]['close'])
                        sell_dates.append(signal_date)
                    else:
                        # 找最近的价格
                        nearest = price_data[price_data['date'] >= signal_date].head(1)
                        if len(nearest) > 0:
                            sell_prices.append(nearest.iloc[0]['close'])
                            sell_dates.append(signal_date)
                
                if len(sell_prices) > 0:
                    fig.add_trace(go.Scatter(
                        x=sell_dates,
                        y=sell_prices,
                        mode='markers',
                        name='卖出',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='red',
                            line=dict(color='darkred', width=2)
                        ),
                        hovertemplate='<b>卖出</b><br>日期: %{x|%Y-%m-%d}<br>价格: %{y:.2f}<extra></extra>'
                    ))
            
            fig.update_layout(
                title=f"{selected_symbol} - 价格与买卖点",
                xaxis_title="日期",
                yaxis_title="价格",
                height=600,
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示统计信息
            col1, col2 = st.columns(2)
            with col1:
                st.metric("买入次数", len(buy_signals))
            with col2:
                st.metric("卖出次数", len(sell_signals))
            
        except Exception as e:
            import traceback
            st.error(f"❌ 加载价格数据时出错: {str(e)}")
            with st.expander("查看详细错误信息"):
                st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # 信号详情表格
    st.markdown("### 📋 信号详情")
    
    # 筛选选项
    col1, col2 = st.columns(2)
    
    with col1:
        action_filter = st.selectbox(
            "筛选信号类型",
            ["全部", "买入", "卖出"]
        )
    
    with col2:
        show_recent = st.slider("显示最近N个信号", 10, 100, 50)
    
    # 应用筛选
    filtered_signals = signals_df.copy()
    
    if action_filter == "买入":
        filtered_signals = filtered_signals[filtered_signals['action'] == 'BUY']
    elif action_filter == "卖出":
        filtered_signals = filtered_signals[filtered_signals['action'] == 'SELL']
    
    # 按日期降序排列
    filtered_signals = filtered_signals.sort_values('date', ascending=False).head(show_recent)
    
    # 格式化显示
    display_signals = filtered_signals.copy()
    display_signals['日期'] = display_signals['date'].dt.strftime('%Y-%m-%d')
    display_signals['行业'] = display_signals['symbol']
    display_signals['操作'] = display_signals['action'].apply(lambda x: '🟢 买入' if x == 'BUY' else '🔴 卖出')
    display_signals['原因'] = display_signals['reason']
    
    # 选择要显示的列
    display_cols = ['日期', '行业', '操作', '原因']
    if 'score' in display_signals.columns:
        display_signals['得分'] = display_signals['score'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
        display_cols.append('得分')
    
    st.dataframe(
        display_signals[display_cols],
        use_container_width=True,
        hide_index=True
    )
    
    # 下载按钮
    csv = signals_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "📥 下载完整信号数据",
        csv,
        "trade_signals.csv",
        "text/csv",
        use_container_width=True
    )


if __name__ == "__main__":
    main()

